# T1_ds_ep020

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T17:58:55Z

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
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '55,90p'"
```
<details><summary>output</summary>

```
    55	Iterate: diagnose the baseline, form a hypothesis, test it on a few frames, measure, and repeat. Work in small,
    56	fast experiments first; you have a fast model, so many short iterations are better than one long attempt.
    57	
    58	## Depth
    59	The baseline's distance from the camera was calibrated against the whole human body and the floor; depth from the
    60	object alone (silhouette scale on an object a few dozen pixels wide, with an estimated K) is far less reliable. You
    61	may correct depth, but only when several independent cues agree (hand contact with the reconstructed human, static
    62	object on the released floor plane, MoGe-2 depth, metric mesh size vs. image size), and quantify their disagreement.
    63	Do not snap the object onto the floor plane: it …[+2440 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -lah /work/inputs /work/outputs 2>/dev/null || true && cat /work/inputs/task.txt && cat /work/inputs/episode.json && cat /work/inputs/camera.json && cat /work/inputs/ground_plane.json'
```
<details><summary>output</summary>

```
/work/inputs:
total 59M
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 17:54 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 21:52 ..
-rw-rw-r-- 1 hanzhou hanzhou  834 Oct 20 17:54 camera.json
-rw-rw-r-- 1 hanzhou hanzhou   66 Oct 20 17:53 episode.json
-rw-rw-r-- 1 hanzhou hanzhou 188K Oct 20 17:54 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou  394 Oct 20 17:53 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou 4.8M Oct 20 17:54 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou  12M Oct 20 17:53 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou  27M Oct 20 17:54 reconstruction.npz
-rw-rw-r-- 1 hanzhou hanzhou  148 Oct 20 17:53 task.txt
-rw-rw-r-- 1 hanzhou hanzhou  17M Oct 20 17:53 video.mp4

/work/outputs:
total 8.0K
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:52 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 21:5 …[+1447 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1847
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1959
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/05_object_pose.json 1772
/pipeline/.stages/06_coconet.json 2828
/pipeline/.stages/07_refinement.json 2131
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/depth_report.json 1789
/pipeline/depth/moge2/aligned.h5 676356140
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 59016
/pipeline/depth/moge2/raw.npy 1957036160
/pipeline/export/episode_000020/edex 561
/pipeline/export/episode_000020/wild_export.json 2798
/pipeline/foundationpose/episode_000020.pkl 160215
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 4231034 …[+548 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction -maxdepth 2 -type f | head -100; echo '---'; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"
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
/code/reconstructio …[+853 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, cv2, json, trimesh, os
r=np.load('/work/inputs/reconstruction.npz')
print('recon keys',r.files)
for k in r.files: print(k,r[k].shape,r[k].dtype)
with h5py.File('/work/inputs/masks.h5') as f:
 print('masks',list(f.keys()))
 for k in f: print(k,f[k].shape,f[k].dtype)
cap=cv2.VideoCapture('/work/inputs/video.mp4'); print('video',cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FPS),cap.get(3),cap.get(4))
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print('scene',s, type(s)); g=s.to_geometry(); print(type(g),g.vertices.shape,g.faces.shape,g.bounds,g.visual.kind)
PY"
```
<details><summary>output</summary>

```
recon keys ['T_world_mesh', 'T_cam_mesh_foundationpose', 'T_cam_mesh_coconet', 'human_vertices_world', 'human_keypoints_world', 'contact_prob_right_left']
T_world_mesh (553, 4, 4) float32
T_cam_mesh_foundationpose (553, 4, 4) float32
T_cam_mesh_coconet (553, 4, 4) float32
human_vertices_world (553, 4610, 3) float32
human_keypoints_world (553, 70, 3) float32
contact_prob_right_left (553, 2) float32
masks ['object', 'person']
object (553, 1152, 1536) bool
person (553, 1152, 1536) bool
video 553.0 30.0 1536.0 1152.0
scene <trimesh.Scene(len(geometry)=1)> <class 'trimesh.scene.scene.Scene'>
<class 'trimesh.base.Trimesh'> (295487, 3) (98533, 3) [[-0.10614258 -0.11585464 -0.12354461]
 [ 0.1142482   0.10137311  0.10902679]] texture

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p /work/scratch /work/outputs && python - <<'PY'
import numpy as np, json, cv2, h5py, trimesh
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'];
t=T[:,:3,3]; Rw=T[:,:3,:3]
# angle increments
rel=np.einsum('fji,fjk->fik',Rw[:-1],Rw[1:])
ang=np.degrees(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
dt=np.linalg.norm(np.diff(t,axis=0),axis=1)
print('T range',t.min(0),t.max(0))
print('center first/last',t[0],t[-1])
for x,name in [(ang,'rot step deg'),(dt,'trans step m')]:
 print(name,'percentiles',np.percentile(x,[0,25,50,75,90,95,99,100]))
 inds=np.argsort(x)[-30:]
 print([(int(i),round(float(x[i]),3)) for i in inds])
# chunks
for a in range(0,553,25):
 b=min(553,a+25); print(a,b,'tmean',np.round(t[a:b].mean(0),3),'tstd',np.round(t[a:b].std(0),3),'rstep',round(float(ang[max(0,a):max(0,b-1)].mean()),2),'tstep',round(float(dt[max(0,a):min(552,b-1)].mean()),4),'contact',np.round(r['contact_prob_right_left'][a:b].mean(0),2))
# reprojection bounds
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=np.asarray(mesh.vertices)
with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][:]; pm=f['person'][:]
areas=om.sum((1,2));
print('object mask area pct',np.percentile(areas,[0,5,25,50,75,95,100]))
for i in [0,50 …[+423 chars]
```
<details><summary>output</summary>

```
T range [0.1481653  0.27243206 3.0369112 ] [0.17003922 0.30466282 3.2334518 ]
center first/last [0.15433478 0.2865271  3.0994813 ] [0.16155978 0.28280914 3.086686  ]
rot step deg percentiles [0.11869406 0.48778763 0.74624243 1.3111943  2.23379266 2.82602534
 4.08095477 4.99810362]
[(347, 2.813), (273, 2.816), (418, 2.838), (321, 2.858), (393, 2.959), (385, 2.963), (399, 2.989), (410, 3.015), (327, 3.041), (344, 3.117), (352, 3.244), (350, 3.283), (348, 3.429), (395, 3.445), (271, 3.508), (398, 3.561), (413, 3.661), (349, 3.665), (343, 3.694), (417, 3.766), (396, 3.778), (397, 3.804), (389, 4.055), (391, 4.072), (342, 4.091), (392, 4.471), (415, 4.744), (329, 4.812), (416, 4.989), (394, 4.998)]
trans step m percentiles [0.00017348 0.00374102 0.0075075  0.01723793 0.04277331 0.06090004
 0.08 …[+2949 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, numpy as np, h5py, os, json, trimesh
from scipy.spatial.transform import Rotation as R
out='/work/scratch/frames'; os.makedirs(out,exist_ok=True)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][:]; pm=f['person'][:]
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=np.asarray(mesh.vertices)
idxs=list(range(0,553,25))+[552]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 obj=np.where(om[i]); per=np.where(pm[i])
 # union bounding around object + person relevant? crop based obj +-120
 y0,x0=np.min(obj,axis=1); y1,x1=np.max(obj,axis=1)
 cx,cy=(x0+x1)//2,(y0+y1)//2
 xa=max(0,cx-180); xb=min(1536,cx+180); ya=max(0,cy-150); yb=min(1152,cy+180)
 vis=im.copy(); vis[pm[i]]=(vis[pm[i]]*0.5+np.array([255,0,255])*0.5).astype(np.uint8)
 vis[om[i]]=(vis[om[i]]*0.5+np.array([0,255,0])*0.5).astype(np.uint8)
 # project mesh vertices
 M=Tcw@T[i]; P=(M[:3,:3]@V.T+M[:3,3:]).T; uv=(K@P.T).T; uv=uv[:,:2]/uv[:,2:3]
 cv2.polylines(vis,[np.round(uv).astype(np.int32)],False,(0,0,255),1)
 crop=vis[ya:yb,xa:xb]; cv2.putText(crop,f'{i}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3); cv2.putText(crop,f'{i}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255 …[+374 chars]
```
<details><summary>output</summary>

```
done 24

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, numpy as np, h5py, os
out='/work/scratch/orig'; os.makedirs(out,exist_ok=True)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][:]
idxs=list(range(0,553,25))+[552]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); yy,xx=np.where(om[i]); cx,cy=int(xx.mean()),int(yy.mean()); crop=im[max(0,cy-75):cy+75,max(0,cx-75):cx+75]; cv2.imwrite(f'{out}/{i:03d}.jpg',crop)
ims=[cv2.imread(f'{out}/{i:03d}.jpg') for i in idxs]
th=[cv2.resize(x,(240,240),interpolation=cv2.INTER_CUBIC) for x in ims]
rows=[]
for j in range(0,len(th),5):
 row=th[j:j+5]
 while len(row)<5: row.append(np.zeros_like(th[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/orig_montage.jpg',np.vstack(rows))
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
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; R=T[:,:3,:3]
# relative rotation to frame 0
rel=np.einsum('ji,fjk->fik',R[0],R)
rv=Rot.from_matrix(rel).as_rotvec(); print('relative angles deg around xyz')
for i in range(0,553,25): print(i,np.round(np.degrees(rv[i]),2),'norm',round(np.linalg.norm(np.degrees(rv[i])),2))
print('abs mean axes?')
# incremental axis
rel2=np.einsum('fji,fjk->fik',R[:-1],R[1:]); rots=Rot.from_matrix(rel2); rv2=rots.as_rotvec(); axis=rv2/np.linalg.norm(rv2,axis=1)[:,None]
for i in range(0,552,25): print(i,'inc',np.round(np.degrees(rv2[i]),2),'axis',np.round(axis[i],2))
# yaw by atan2 local x z vector world
# project object x axis remove y
for i in range(0,553,25):
 x=R[i,:,0]; z=R[i,:,2]; print('orient',i, np.round(x,2),np.round(z,2),'yaw',round(np.degrees(np.arctan2(x[2],x[0])),2))
PY"
```
<details><summary>output</summary>

```
relative angles deg around xyz
0 [0. 0. 0.] norm 0.0
25 [ 0.02 -0.29 -0.19] norm 0.35
50 [0.05 0.7  1.03] norm 1.25
75 [-0.2   1.47  0.93] norm 1.75
100 [-0.6   1.65  1.21] norm 2.14
125 [-0.26  0.22  0.41] norm 0.53
150 [0.58 0.53 0.07] norm 0.79
175 [0.38 0.47 0.4 ] norm 0.73
200 [0.31 0.8  1.  ] norm 1.31
225 [ 0.48  1.1  19.63] norm 19.66
250 [-0.06  1.54 53.41] norm 53.43
275 [  1.87   2.77 102.01] norm 102.06
300 [  2.3    3.44 102.05] norm 102.14
325 [  2.03   2.8  110.83] norm 110.88
350 [  1.39   3.12 163.03] norm 163.07
375 [  -1.02   -2.59 -154.93] norm 154.96
400 [ -2.05  -0.97 -95.14] norm 95.17
425 [ -1.7   -0.41 -51.25] norm 51.28
450 [ -0.2    0.23 -51.14] norm 51.14
475 [ -1.25  -0.61 -51.84] norm 51.86
500 [ -0.87  -0.4  -51.  ] norm 51.01
525 [  0.4   -2.45 -51.12] norm  …[+2604 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "du -sh /work/scratch /work/tools 2>/dev/null; df -h /work; python - <<'PY'
import h5py
for p in ['/pipeline/inference/materialized_inputs.h5','/pipeline/work/masks_k0.h5']:
 print('\\n',p)
 with h5py.File(p) as f:
  def vis(name,obj):
   if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype)
  f.visititems(vis)
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/materialized_inputs.h5
K_rois (553, 3, 3) float32
bboxes (553, 4) float32
identity_json () object
input_rgbs (553, 3, 224, 224) float32
input_xyz (553, 6, 224, 224) float32
observed_human_masks (553, 224, 224) bool
observed_object_masks (553, 224, 224) bool
render_rgbs (553, 3, 224, 224) float32
render_xyz (553, 6, 224, 224) float32

 /pipeline/work/masks_k0.h5
episode_000020/000000-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000020/000000-k0.person_mask.png (1152, 1536) uint8
episode_000020/000001-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000020/000001-k0.person_mask.png (1152, 1536) uint8
episode_000020/000002-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000020/000002-k0.person_mask.png (1152, 1536) uint8
episode_000020/000003-k0.obj_rend_mask.png (1152 …[+67052 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['torch','torchvision','pytorch3d','nvdiffrast','open3d','scipy','skimage','h5py','trimesh','cv2','kornia']
for m in mods:
 try:
  x=__import__(m); print(m,getattr(x,'__version__','?'))
 except Exception as e: print(m,'NO',e)
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124
torchvision 0.20.1+cu124
pytorch3d 0.7.9
nvdiffrast 0.4.0
open3d 0.20.0
scipy 1.16.3
skimage 0.26.0
h5py 3.16.0
trimesh 5.1.1
cv2 4.11.0
kornia 0.8.3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False); g=s.to_geometry(); v=g.visual
print(type(v),v.kind)
print(v.__dict__.keys())
for k,val in v.__dict__.items():
 try: print(k, type(val), getattr(val,'shape',None), getattr(val,'dtype',None), val if np.size(val)<20 else '')
 except: pass
print('uv',g.visual.uv.shape if hasattr(g.visual,'uv') else None)
print('material',g.visual.material)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'> texture
dict_keys(['vertex_attributes', '_cache', 'material', 'face_materials', 'mesh'])
vertex_attributes <class 'trimesh.caching.DataStore'> None None <trimesh.caching.DataStore object at 0x7f9187e22a90>
material <class 'trimesh.visual.material.PBRMaterial'> None None <trimesh.visual.material.PBRMaterial object at 0x7f9187e23ad0>
face_materials <class 'NoneType'> None None None
mesh <class 'trimesh.base.Trimesh'> None None <trimesh.Trimesh(vertices.shape=(295487, 3), faces.shape=(98533, 3))>
uv (295487, 2)
material <trimesh.visual.material.PBRMaterial object at 0x7f9187e23ad0>

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
mat=m.visual.material
print(mat)
for attr in ['baseColorTexture','baseColorFactor','metallicFactor','roughnessFactor','emissiveTexture','normalTexture']:
 x=getattr(mat,attr,None); print(attr,type(x),getattr(x,'shape',None),getattr(x,'mode',None))
 if x is not None and hasattr(x,'getdata'): print(np.array(x).shape, np.array(x).min(),np.array(x).max())
PY"
```
<details><summary>output</summary>

```
<trimesh.visual.material.PBRMaterial object at 0x77f217d0ed10>
baseColorTexture <class 'PIL.Image.Image'> None RGB
(2344, 2320, 3) 0 255
baseColorFactor <class 'NoneType'> None None
metallicFactor <class 'float'> None None
roughnessFactor <class 'NoneType'> None None
emissiveTexture <class 'NoneType'> None None
normalTexture <class 'NoneType'> None None

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_test.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py, nvdiffrast.torch as dr, time, os
from PIL import Image
torch.manual_seed(0)
dev='cuda'
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev)
F=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=dev)
tex=np.array(mesh.visual.material.baseColorTexture)
tex=torch.tensor(tex,dtype=torch.float32,device=dev)/255
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32); Twc=np.array(cam['T_world_cam'],dtype=np.float32); Tcw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
H,W=1152,1536
gl=dr.RasterizeCudaContext()
def render(Tm, twc=Twc, k=K):
    # mesh -> world -> cam (torch)
    TT=torch.tensor(Tm,device=dev,dtype=torch.float32)
    vv=V@TT[:3,:3].T+TT[:3,3]
    vv=vv@torch.tensor(twc[:3,:3],device=dev).T+torch.tensor(twc[:3,3],device=dev)
    z=vv[:,2]
    fx,fy=k[0,0],k[1,1]; cx,cy=k[0,2],k[1,2]
    xn=2*(fx*vv[:,0]/z+cx)/W-1
    yn=1-2*(fy*vv[:,1]/z+cy)/H
    zn=(z-0.1)/(20-0.1)*2-1
    pos=torch.stack([xn,yn,zn,torch.ones_like(z)],1)[None]
    rast,gb_uv,gb_tri=dr.rasterize(gl,pos,F,resolution=[H,W])
    uv, _=dr.interpolate(UV[None],rast,F)
    # sample texture normalized
    col=torch.nn.functional.grid_sample(tex.permute(2,0,1)[None],(uv*2- …[+1204 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 37, in <module>
    m,col,dep=render(T[i]); m=m.cpu().numpy(); col=(col.cpu().numpy()*255).astype(np.uint8)
              ^^^^^^^^^^^^
  File "/work/scratch/render_test.py", line 26, in render
    rast,gb_uv,gb_tri=dr.rasterize(gl,pos,F,resolution=[H,W])
    ^^^^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 3, got 2)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/rast,gb_uv,gb_tri=dr.rasterize/rast,gb_tri=dr.rasterize/' /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 37, in <module>
    m,col,dep=render(T[i]); m=m.cpu().numpy(); col=(col.cpu().numpy()*255).astype(np.uint8)
              ^^^^^^^^^^^^
  File "/work/scratch/render_test.py", line 31, in render
    dep, _=dr.interpolate(z[None,:,None],rast,F)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/py …[+286 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/dr.interpolate(z\\[None,:,None\\]/dr.interpolate(z[None,:,None].contiguous()/\" /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 iou 0.0 gt 3396 pred 3208 inter 0
50 iou 0.0 gt 3441 pred 3209 inter 0
100 iou 0.0 gt 3335 pred 3081 inter 0
150 iou 0.0 gt 3315 pred 3099 inter 0
200 iou 0.0 gt 3266 pred 3119 inter 0
250 iou 0.0 gt 3182 pred 2940 inter 0
300 iou 0.0 gt 3458 pred 3274 inter 0
350 iou 0.0 gt 3401 pred 3119 inter 0
400 iou 0.0 gt 3248 pred 3127 inter 0
450 iou 0.0 gt 3508 pred 3367 inter 0
500 iou 0.0 gt 3583 pred 3473 inter 0
552 iou 0.0 gt 3514 pred 3508 inter 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/yn=1-2\\*(fy\\*vv\\[:,1\\]\\/z+cy)\\/H/yn=2*(fy*vv[:,1]\\/z+cy)\\/H-1/' /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 iou 0.0 gt 3396 pred 3209 inter 0
50 iou 0.0 gt 3441 pred 3206 inter 0
100 iou 0.0 gt 3335 pred 3081 inter 0
150 iou 0.0 gt 3315 pred 3098 inter 0
200 iou 0.0 gt 3266 pred 3121 inter 0
250 iou 0.0 gt 3182 pred 2937 inter 0
300 iou 0.0 gt 3458 pred 3274 inter 0
350 iou 0.0 gt 3401 pred 3119 inter 0
400 iou 0.0 gt 3248 pred 3125 inter 0
450 iou 0.0 gt 3508 pred 3365 inter 0
500 iou 0.0 gt 3583 pred 3471 inter 0
552 iou 0.0 gt 3514 pred 3506 inter 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as f:
 for i in [0,250,552]:
  y,x=np.where(f['object'][i]); print(i,x.mean(),y.mean(),x.min(),x.max(),y.min(),y.max())
PY"
```
<details><summary>output</summary>

```
0 755.8103651354535 609.1637220259129 726 785 576 640
250 753.6761755485893 609.209090909091 725 782 578 640
552 755.6841206602163 607.056061468412 723 786 574 638

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# modify print pred bbox using import? rerun quick import without loop impossible. patch existing to print bbox and run frame only
s=open('/work/scratch/render_test.py').read(); s=s.replace(\"for i in [0,50,100,150,200,250,300,350,400,450,500,552]:\",\"for i in [0]:\")
s=s.replace(\"print(i,'iou'\", \"print('predbbox',np.where(m)[1].min(),np.where(m)[1].max(),np.where(m)[0].min(),np.where(m)[0].max()); print(i,'iou'\")
open('/work/scratch/render_one.py','w').write(s)
PY
python /work/scratch/render_one.py"
```
<details><summary>output</summary>

```
predbbox 813 870 634 698
0 iou 0.0 gt 3396 pred 3209 inter 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh,h5py
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']; c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);Tcw=np.linalg.inv(W)
M=T[0]; p=M[:3,3]; pc=Tcw[:3,:3]@p+Tcw[:3,3];print('p world',p,'cam',pc,'uv',K@pc/(pc[2]))
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=g.vertices; M=Tcw@T[0]; P=(M[:3,:3]@V.T).T+M[:3,3];uv=(K@P.T).T;uv=uv[:,:2]/uv[:,2:]
print(uv.min(0),uv.max(0),uv.mean(0))
with h5py.File('/work/inputs/masks.h5') as f:
 y,x=np.where(f['object'][0]);print(x.min(),x.max(),y.min(),y.max(),x.mean(),y.mean())
PY"
```
<details><summary>output</summary>

```
p world [0.15433478 0.2865271  3.0994813 ] cam [-0.0041333   0.16949385  3.04945983] uv [756.98852704 606.07695687   1.        ]
[729.20125626 572.91394037] [788.16718252 638.55155291] [754.41996465 607.065907  ]
726 785 576 640 755.8103651354535 609.1637220259129

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/m,col,dep=render(T\\[i\\])/m,col,dep=render(T[i],Tcw)/\" /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 iou 0.8687205106855398 gt 3396 pred 3337 inter 3130
50 iou 0.8702986725663717 gt 3441 pred 3322 inter 3147
100 iou 0.8863833477883781 gt 3335 pred 3190 inter 3066
150 iou 0.9059304703476483 gt 3315 pred 3209 inter 3101
200 iou 0.8953657825706791 gt 3266 pred 3237 inter 3072
250 iou 0.864897107068297 gt 3182 pred 3071 inter 2900
300 iou 0.8990038738240177 gt 3458 pred 3405 inter 3249
350 iou 0.8896159317211949 gt 3401 pred 3241 inter 3127
400 iou 0.9127218934911243 gt 3248 pred 3217 inter 3085
450 iou 0.8833780160857909 gt 3508 pred 3517 inter 3295
500 iou 0.9172432144757849 gt 3583 pred 3622 inter 3447
552 iou 0.8903497239021825 gt 3514 pred 3675 inter 3386

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_hyps.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, nvdiffrast.torch as dr, time
torch.manual_seed(0); dev='cuda'; H,W=1152,1536
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev); Fc=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32); Kp=np.array(cam['K_pipeline'],dtype=np.float32); Tcw=np.linalg.inv(np.array(cam['T_world_cam'])).astype(np.float32)
r=np.load('/work/inputs/reconstruction.npz'); Ts=[r['T_world_mesh'],r['T_cam_mesh_foundationpose'],r['T_cam_mesh_coconet']]; ks=[K,Kp,Kp]
gl=dr.RasterizeCudaContext()
def masks_batch(Tb,k,n=1):
 # Tb n,4,4 camera final
 TT=torch.tensor(Tb,device=dev,dtype=torch.float32)
 vv=torch.einsum('nij,vj->nvi',TT[:,:3,:3],V)+TT[:,None,:3,3]
 z=vv[:,:,2]; fx,fy=k[0,0],k[1,1];cx,cy=k[0,2],k[1,2]
 xn=2*(fx*vv[:,:,0]/z+cx)/W-1; yn=2*(fy*vv[:,:,1]/z+cy)/H-1; zn=(z-0.1)/(20-0.1)*2-1
 pos=torch.stack([xn,yn,zn,torch.ones_like(z)],2).contiguous()
 rast,_=dr.rasterize(gl,pos,Fc,resolution=[H,W])
 return rast[:,:,:,3]>0
with h5py.File('/work/inputs/masks.h5') as f: gt=f['object'][:]
# frame0 world to camera
Twc=np.array(cam['T_world_cam'],dtype=np.float32); Tcw4=np.linalg.inv(Twc)
Tbase=np.einsum('ij,fjk->fik',Tcw4,Ts[0])
allH=[Tbase,Ts[1],Ts[2]]
out=[]
t0=time.time()
for s in range(553):
 bb=[]
 for hi,T in  …[+661 chars]
```
<details><summary>output</summary>

```
time 20.38305401802063
base iou pct [0.82965821 0.85729911 0.87466667 0.88982604 0.90266742 0.91444319
 0.92202729] mean 0.8882399506648934
fp iou pct [0.83772422 0.87261115 0.90305701 0.92513677 0.94690782 0.96877209
 0.97414806] mean 0.9236361409498223
coco iou pct [0.82777926 0.85468583 0.87330191 0.88719761 0.90045113 0.91325223
 0.9230545 ] mean 0.8859797397100385
0 25 [[ 8.76000e-01  1.96900e+00 -2.15100e+00  3.38356e+03]
 [ 9.00000e-01 -9.00000e-03 -7.38000e-01  3.07272e+03]
 [ 8.69000e-01  1.90600e+00 -2.23700e+00  3.16860e+03]]
25 50 [[ 8.93000e-01  1.96600e+00 -1.35500e+00  3.36592e+03]
 [ 9.34000e-01  8.00000e-03 -2.90000e-01  3.20768e+03]
 [ 8.87000e-01  1.92200e+00 -1.42000e+00  3.21296e+03]]
50 75 [[ 8.68000e-01  2.52700e+00 -1.59300e+00  3.27348e+03]
 [ 9.32000e-01 -1.48000e …[+3578 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; import json
c=json.load(open('/work/inputs/camera.json'));Tcw=np.linalg.inv(np.array(c['T_world_cam'])); base=np.einsum('ij,fjk->fik',Tcw,T); R0=base[:,:3,:3]
for name,key in [('fp','T_cam_mesh_foundationpose'),('coco','T_cam_mesh_coconet')]:
 RR=r[key][:,:3,:3]; rel=np.einsum('fji,fjk->fik',R0,RR); ang=np.degrees(R.from_matrix(rel).magnitude());
 print(name,'diff pct',np.percentile(ang,[0,5,25,50,75,95,100]))
 for i in range(0,553,25): print(i,round(float(ang[i]),1),end='; ')
 print()
# translation differences
 bt=base[:,:3,3]
 for name,key in [('fp','T_cam_mesh_foundationpose'),('coco','T_cam_mesh_coconet')]: print(name,'tdiff',np.linalg.norm(bt-r[key][:,:3,3],axis=1).mean())
PY"
```
<details><summary>output</summary>

```
fp diff pct [0.75422067 1.51024584 2.51359465 2.97814901 3.34812756 3.94324334
 4.81241468]
0 3.7; 25 2.6; 50 3.3; 75 3.9; 100 3.1; 125 2.4; 150 2.4; 175 3.0; 200 3.5; 225 3.0; 250 3.3; 275 2.1; 300 1.8; 325 1.4; 350 3.4; 375 2.2; 400 2.1; 425 3.0; 450 2.9; 475 3.0; 500 3.2; 525 2.6; 550 2.6; 
fp tdiff 0.1263777392040073
coco tdiff 0.10948896674645572
coco diff pct [1.30006387 1.30122799 1.30203682 1.30281195 1.30328871 1.30374286
 1.30428633]
0 1.3; 25 1.3; 50 1.3; 75 1.3; 100 1.3; 125 1.3; 150 1.3; 175 1.3; 200 1.3; 225 1.3; 250 1.3; 275 1.3; 300 1.3; 325 1.3; 350 1.3; 375 1.3; 400 1.3; 425 1.3; 450 1.3; 475 1.3; 500 1.3; 525 1.3; 550 1.3; 
fp tdiff 0.1263777392040073
coco tdiff 0.10948896674645572

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/pipeline_report.json; echo; cat /pipeline/.stages/07_refinement.json; echo; cat /pipeline/.stages/06_coconet.json'
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
    "mtime_ns": 1791236187379682281,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000020/inference/episode_000020_step200000_before_after.mp4",
    "size": 38954331
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frame …[+10079 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline/export/episode_000020 -type f -maxdepth 4 -printf '%p %s\\n'; cat /pipeline/export/episode_000020/wild_export.json"
```
<details><summary>output</summary>

```
/pipeline/export/episode_000020/human_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000020/human_masks/front_stereo_camera_left.h5 13297584
/pipeline/export/episode_000020/images/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000020/images/front_stereo_camera_left.h5 506552992
/pipeline/export/episode_000020/object_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000020/object_masks/front_stereo_camera_left.h5 11434410
/pipeline/export/episode_000020/object_mesh/output_aligned.glb 11571468
/pipeline/export/episode_000020/edex 561
/pipeline/export/episode_000020/wild_export.json 2798
{
  "aligned_object_extents_m": [
    0.21707800030708313,
    0.22033025324344635,
    0.2319326251745224
  ],
 …[+2686 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"T_world_mesh\\|depth correction\\|depth_correction\\|T_cam_mesh_coconet\" -n /code/reconstruction /workspace 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_gsplat_refinement/lib/pose_fields.py:206:        # Per-track multiplicative depth correction from align_hands. The
/code/reconstruction/modules/v2d_gsplat_refinement/lib/pose_fields.py:306:        # Multiplicative depth correction (hand_scale) — applied around the
/code/reconstruction/modules/v2d_gsplat_refinement/lib/pose_fields.py:360:        # Multiplicative depth correction (hand_scale), per-batch-elt-centred.
/code/reconstruction/modules/v2d_gsplat_refinement/lib/io.py:149:    hand_scale:    float        -- per-track multiplicative depth correction
/code/reconstruction/modules/v2d_hamer/lib/render_hands_aligned_video.py:179:    # Apply the per-track multiplicative depth correction emitted by
/code/reconstruction/modules/v2d_hamer/lib/align_hands.py:275 …[+78 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); uv=m.visual.uv
print(uv.min(0),uv.max(0),np.percentile(uv,[1,5,50,95,99],axis=0))
im=np.array(m.visual.material.baseColorTexture); Image.fromarray(im).resize((580,586)).save('/work/scratch/texture.jpg')
# crop texture content bbox nonwhite
print(im.mean((0,1)),np.unique(im.reshape(-1,3),axis=0).shape)
PY"
```
<details><summary>output</summary>

```
[0.002088   0.00292897] [0.99717498 0.99791902] [[0.026802   0.01814699]
 [0.0776202  0.04380202]
 [0.499944   0.33933997]
 [0.941266   0.91227901]
 [0.97611804 0.96934599]]
[108.96198861 113.63378049 110.36030272] (92696, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];R=T[:,:3,:3]
# yaw local x axis in horizontal world xz, but baseline object axis local z->world y. use atan2(R[2,0],R[0,0])
yaw=np.unwrap(np.arctan2(R[:,2,0],R[:,0,0]))
yaw=np.degrees(yaw-yaw[0])
# smooth derivative 5 frame
for i in range(0,553,10): print(i, round(yaw[i],2), round((yaw[min(552,i+10)]-yaw[i])/10,3), np.round(T[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
0 0.0 -0.055 [0.154 0.287 3.099]
10 -0.55 0.028 [0.155 0.284 3.079]
20 -0.27 -0.009 [0.153 0.285 3.085]
30 -0.36 -0.07 [0.154 0.289 3.09 ]
40 -1.06 0.002 [0.155 0.292 3.125]
50 -1.04 0.096 [0.155 0.288 3.093]
60 -0.08 -0.003 [0.156 0.289 3.116]
70 -0.11 -0.144 [0.157 0.292 3.136]
80 -1.55 0.013 [0.163 0.295 3.147]
90 -1.43 0.019 [0.159 0.299 3.159]
100 -1.24 0.061 [0.16  0.301 3.155]
110 -0.63 -0.039 [0.159 0.303 3.172]
120 -1.02 0.019 [0.157 0.299 3.152]
130 -0.82 0.035 [0.159 0.298 3.156]
140 -0.47 0.038 [0.158 0.298 3.141]
150 -0.09 -0.047 [0.158 0.301 3.159]
160 -0.56 0.009 [0.157 0.298 3.137]
170 -0.46 -0.016 [0.158 0.3   3.163]
180 -0.62 -0.064 [0.157 0.299 3.142]
190 -1.26 0.024 [0.16  0.302 3.167]
200 -1.02 -0.12 [0.158 0.301 3.144]
210 -2.22 -1.173 [0.158 0.302 3.151]
220 -13.95 - …[+1294 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/render_test.py').read()
s=s.replace(\"for i in [0,50,100,150,200,250,300,350,400,450,500,552]:\",\"for i in [0,250,400,552]:\")
s=s.replace(\"out=np.hstack([im[y0:y1,x0:x1],blend[y0:y1,x0:x1],col[y0:y1,x0:x1]])\", \"out=np.hstack([im[y0:y1,x0:x1],blend[y0:y1,x0:x1],col[y0:y1,x0:x1]])\\n cv2.imwrite(f'/work/scratch/tex_{i:03d}.png',cv2.cvtColor(col[y0:y1,x0:x1],cv2.COLOR_RGB2BGR))\")
open('/work/scratch/render_tex.py','w').write(s)
PY
python /work/scratch/render_tex.py"
```
<details><summary>output</summary>

```
0 iou 0.8687205106855398 gt 3396 pred 3337 inter 3130
250 iou 0.864897107068297 gt 3182 pred 3071 inter 2900
400 iou 0.9127218934911243 gt 3248 pred 3217 inter 3085
552 iou 0.8903497239021825 gt 3514 pred 3675 inter 3386

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mat=m.visual.material
print(mat.__dict__)
PY"
```
<details><summary>output</summary>

```
{'_data': {'metallicFactor': 0.0, 'baseColorTexture': <PIL.Image.Image image mode=RGB size=2320x2344 at 0x7C08DE77BA10>, 'doubleSided': True, 'name': '_texture'}}

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/score_yaw.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, nvdiffrast.torch as dr, cv2
from scipy.spatial.transform import Rotation as Rot
torch.manual_seed(0); dev='cuda'
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev); Fc=torch.tensor(mesh.faces,dtype=torch.int32,device=dev); UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=dev)
tex=torch.tensor(np.array(mesh.visual.material.baseColorTexture),dtype=torch.float32,device=dev).permute(2,0,1)[None]/255
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32); Tcw=np.linalg.inv(np.array(cam['T_world_cam'],dtype=np.float32))
r=np.load('/work/inputs/reconstruction.npz'); Ts=r['T_world_mesh']
cap=cv2.VideoCapture('/work/inputs/video.mp4'); with h5py.File('/work/inputs/masks.h5') as f:oall=f['object'][:];pall=f['person'][:]
gl=dr.RasterizeCudaContext()
def render(T, i, S=160):
 im0=None
 # crop center
 yy,xx=np.where(oall[i]); cx,cy=int(xx.mean()),int(yy.mean()); x0,y0=cx-S//2,cy-S//2
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); im=cv2.cvtColor(im[:,x0:x0+S,y0:y0+S],cv2.COLOR_BGR2RGB)
 TT=torch.tensor(T,device=dev,dtype=torch.float32); vv=V@TT[:3,:3].T+TT[:3,3]; vv=vv@torch.tensor(Tcw[:3,:3],device=dev).T+torch.tensor(Tcw[:3,3],device=dev)
 z=vv[:,2]; k=K.copy(); k[0,2]-=x0;k[1,2]-=y0; W=H=S; fx,fy=k[0,0],k[1,1];px=k[0,2];py=k[1,2]
 x …[+1892 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/score_yaw.py", line 8
    cap=cv2.VideoCapture('/work/inputs/video.mp4'); with h5py.File('/work/inputs/masks.h5') as f:oall=f['object'][:];pall=f['person'][:]
                                                    ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/score_yaw.py';s=open(p).read();s=s.replace(\"cap=cv2.VideoCapture('/work/inputs/video.mp4'); with h5py.File('/work/inputs/masks.h5') as f:oall=f['object'][:];pall=f['person'][:]\",\"cap=cv2.VideoCapture('/work/inputs/video.mp4')\\nwith h5py.File('/work/inputs/masks.h5') as f:\\n oall=f['object'][:];pall=f['person'][:]\")
s=s.replace(\"  Tc=RR.copy();Tc[:3,3]=c-RR@c\\n  TT=T.copy();TT[:3,:3]=RR@T[:3,:3];TT[:3,3]=Tc[:3,3]+RR@(T[:3,3]-c) # wrong no\\n  TT[:3,3]=c+RR@(T[:3,3]-c)\",\"  TT=T.copy();TT[:3,:3]=RR@T[:3,:3];TT[:3,3]=c+RR@(T[:3,3]-c)\")
open(p,'w').write(s)
PY
python /work/scratch/score_yaw.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/score_yaw.py", line 40, in <module>
    T=Ts[i]; im,m,col,x0,y0=render(T,i,S=160)
                            ^^^^^^^^^^^^^^^^^
  File "/work/scratch/score_yaw.py", line 16, in render
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); im=cv2.cvtColor(im[:,x0:x0+S,y0:y0+S],cv2.COLOR_BGR2RGB)
                                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) /io/opencv/modules/imgproc/src/color.simd_helpers.hpp:92: error: (-15:Bad number of channels) in function 'cv::impl::{anonymous}::CvtHelper<VScn, VDcn, VDepth, sizePolicy>::CvtHelper(cv::InputArray, cv::OutputArray, int) [with VScn = cv::impl::{anonymous}::Set<3, 4>; VDcn = cv::impl::{anonymous}::Set<3 …[+312 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/im\\[:,x0:x0+S,y0:y0+S\\]/im[y0:y0+S,x0:x0+S]/\" /work/scratch/score_yaw.py
python /work/scratch/score_yaw.py"
```
<details><summary>output</summary>

```
FRAME 0
Traceback (most recent call last):
  File "/work/scratch/score_yaw.py", line 47, in <module>
    sc=scores(im,m,col)
       ^^^^^^^^^^^^^^^^
  File "/work/scratch/score_yaw.py", line 27, in scores
    valid=m & ~pall[i][y0:y0+S,x0:x0+S]
                             ^
NameError: name 'S' is not defined

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/valid=m & ~pall\\[i\\]\\[y0:y0+S,x0:x0+S\\]/valid=m \\& ~pall[i][y0:y0+m.shape[0],x0:x0+m.shape[1]]/' /work/scratch/score_yaw.py
python /work/scratch/score_yaw.py"
```
<details><summary>output</summary>

```
FRAME 0
Traceback (most recent call last):
  File "/work/scratch/score_yaw.py", line 47, in <module>
    sc=scores(im,m,col)
       ^^^^^^^^^^^^^^^^
  File "/work/scratch/score_yaw.py", line 34, in scores
    ga=cv2.Laplacian(a,cv2.CV_32F);gb=cv2.Laplacian(b,cv2.CV_32F); ga=ga[valid];gb=gb[valid]; gn=(ga*gb).mean()/(ga.std()*gb.std()+1e-6)
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) /io/opencv/modules/imgproc/src/filter.simd.hpp:3187: error: (-215:Assertion failed) cn == CV_MAT_CN(dstType) && ddepth >= sdepth in function 'getLinearFilter'


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/astype(float)/astype(np.float32)/g' /work/scratch/score_yaw.py
python /work/scratch/score_yaw.py"
```
<details><summary>output</summary>

```
FRAME 0
-180 iou 0.875 scores [ 1.8200e-01 -2.2000e-02  4.5000e-02  9.3163e+01]
-135 iou 0.839 scores [3.6000e-02 2.1000e-02 2.7000e-02 7.5478e+01]
-90 iou 0.846 scores [ 2.0000e-02 -4.0000e-02  2.0000e-02  7.6065e+01]
-45 iou 0.842 scores [4.3000e-02 4.0000e-03 2.2000e-02 7.5883e+01]
0 iou 0.869 scores [2.6000e-02 1.1000e-02 3.7000e-02 7.0962e+01]
45 iou 0.879 scores [ 1.4000e-02 -3.0000e-03  4.0000e-02  6.8883e+01]
90 iou 0.851 scores [-6.7000e-02 -1.3000e-02  4.0000e-02  9.3258e+01]
135 iou 0.827 scores [-7.00000e-03 -3.70000e-02  8.00000e-03  1.00564e+02]
180 iou 0.875 scores [ 1.8200e-01 -2.2000e-02  4.5000e-02  9.3163e+01]
FRAME 50
-180 iou 0.887 scores [2.3600e-01 1.0000e-02 6.2000e-02 9.0481e+01]
-135 iou 0.848 scores [ 2.8000e-02 -1.3000e-02  2.5000e-02  7.3952e+01]
-90 iou 0.852  …[+3034 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/yaw_vis.py <<'PY'
exec(open('/work/scratch/score_yaw.py').read().split('for i in [0,50,100,460,500,552]:')[0])
from scipy.spatial.transform import Rotation as Rot
for i in [0,250,400,552]:
 im,m,col,x0,y0=render(Ts[i],i,S=130)
 tiles=[]
 orig=im.copy(); orig[~oall[i][y0:y0+130,x0:x0+130]]=(orig[~oall[i][y0:y0+130,x0:x0+130]]*.3).astype(np.uint8)
 tiles.append(orig)
 for off in [-180,-90,45,0,90,180]:
  c=Ts[i][:3,3];RR=Rot.from_rotvec([0,np.deg2rad(off),0]).as_matrix().astype(np.float32);TT=Ts[i].copy();TT[:3,:3]=RR@TT[:3,:3];TT[:3,3]=c
  im2,m2,col2,_,_=render(TT,i,S=130); comp=im2.copy(); comp[m2]=(0.25*comp[m2]+0.75*col2[m2]).astype(np.uint8);cv2.putText(comp,str(off),(4,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),2);tiles.append(comp)
 out=np.hstack(tiles);cv2.imwrite(f'/work/scratch/yawvis_{i}.png',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
PY
python /work/scratch/yaw_vis.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s/(uv\\*2-1).flip(-1)/(uv*2-1)/\" /work/scratch/yaw_vis.py > /work/scratch/yaw_vis_noflip.py
# modify output names to avoid overwrite
sed -i \"s/yawvis_/yawvisnf_/\" /work/scratch/yaw_vis_noflip.py
python /work/scratch/yaw_vis_noflip.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/yaw_vis_noflip.py';s=open(p).read();s=s.replace(\"exec(open('/work/scratch/score_yaw.py').read().split('for i in [0,50,100,460,500,552]:')[0])\",\"src=open('/work/scratch/score_yaw.py').read().split('for i in [0,50,100,460,500,552]:')[0].replace('(uv*2-1).flip(-1)','(uv*2-1)')\\nexec(src)\");open(p,'w').write(s)
PY
python /work/scratch/yaw_vis_noflip.py"
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
r=np.load('/work/inputs/reconstruction.npz');K=r['human_keypoints_world'];T=r['T_world_mesh'];cp=r['contact_prob_right_left'][:,0]
# inspect hand points frame 300
for i in [0,150,250,300,350,400,450]:
 k=K[i,21:42];print(i,'hand range',np.round(k.min(0),3),np.round(k.max(0),3),'mean',np.round(k.mean(0),3),'contact',cp[i])
PY"
```
<details><summary>output</summary>

```
0 hand range [-0.421  0.347  2.952] [-0.356  0.484  3.028] mean [-0.387  0.435  2.994] contact 3.5475287e-10
150 hand range [0.087 0.104 3.089] [0.204 0.213 3.165] mean [0.159 0.165 3.119] contact 2.1175223e-10
250 hand range [0.091 0.119 3.168] [0.213 0.22  3.238] mean [0.164 0.177 3.199] contact 9.304805e-09
300 hand range [0.087 0.108 3.094] [0.207 0.218 3.176] mean [0.161 0.172 3.13 ] contact 3.69294e-08
350 hand range [0.1   0.097 3.119] [0.214 0.204 3.199] mean [0.173 0.158 3.155] contact 1.867875e-08
400 hand range [0.095 0.113 3.096] [0.202 0.213 3.169] mean [0.162 0.17  3.128] contact 2.666405e-09
450 hand range [0.061 0.113 3.115] [0.196 0.199 3.171] mean [0.143 0.161 3.141] contact 5.2601083e-09

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');cp=r['contact_prob_right_left'];print(cp.min(0),cp.max(0),np.percentile(cp,[0,25,50,75,95,100],axis=0));
for i in range(100,500,10): print(i,cp[i])
PY"
```
<details><summary>output</summary>

```
[1.5426044e-14 1.9589911e-12] [2.6090055e-07 1.0000000e+00] [[1.54260438e-14 1.95899113e-12]
 [9.66142305e-11 5.32786430e-07]
 [5.26089838e-10 9.67267036e-01]
 [4.04274614e-09 9.99952793e-01]
 [4.56270911e-08 9.99999666e-01]
 [2.60900549e-07 1.00000000e+00]]
100 [1.9086775e-10 4.9867333e-05]
110 [2.3555011e-09 1.8219704e-05]
120 [1.5631256e-08 4.5320235e-06]
130 [9.3683976e-09 3.9789863e-03]
140 [1.0742841e-09 9.7781497e-01]
150 [2.1175223e-10 9.9936450e-01]
160 [4.8782944e-10 9.9999905e-01]
170 [1.4034569e-10 9.9999976e-01]
180 [1.158907e-09 9.999965e-01]
190 [1.2780430e-09 9.9928254e-01]
200 [3.8930262e-10 9.9996936e-01]
210 [8.3687357e-10 9.9997365e-01]
220 [1.2882923e-10 9.9999970e-01]
230 [5.4363003e-10 9.9980980e-01]
240 [7.6487389e-10 9.9969786e-01]
250 [9.3048049e-09 9.9999535e-01] …[+817 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']; Kp=r['human_keypoints_world']; cp=r['contact_prob_right_left'][:,1]
# center of palm keypoint guesses; print point index movement and relative vector
for idx in [0,1,5,9,13,17]:
 h=Kp[:,21:42][:,idx]; d=T[:,:3,3]-h
 print('idx',idx,'h std',h.std(0),'rel mean',d[160:280].mean(0),'rel std',d[160:280].std(0),'rel step',np.linalg.norm(np.diff(d[160:280],axis=0),axis=1).mean())
# object center vs mean hand
h=Kp[:,21:42].mean(1);d=T[:,:3,3]-h
print('mean hand')
for a,b in [(0,120),(140,205),(205,280),(280,320),(320,420),(420,552)]:
 print(a,b,'hstd',h[a:b].std(0),'rel mean',d[a:b].mean(0),'rel std',d[a:b].std(0),'t std',T[a:b,:3,3].std(0))
# smoothness of hand translations
for a,b in [(140,450),(160,280),(320,420)]:
 print('smooth',a,b,'h step',np.linalg.norm(np.diff(h[a:b],axis=0),axis=1).mean(),'obj step',np.linalg.norm(np.diff(T[a:b,:3,3],axis=0),axis=1).mean())
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=m.vertices
# per frame min/max world y and pc z
Ys=[]
for i in range(553):
 p=V@T[i,:3,:3].T+T[i,:3,3];Ys.append([p[:,1].min(),p[:,1].max(),p[:,2].mean()])
Ys=np.array(Ys)
for i in range(0,553,20): print(i,np.round(Ys[i],4))
print('stats ymax pct',np.percentile(Ys[:,1],[0,5,50,95,100]))
PY"
```
<details><summary>output</summary>

```
0 [0.1631 0.3989 3.1194]
20 [0.1618 0.3977 3.1049]
40 [0.1685 0.4042 3.1455]
60 [0.1657 0.4009 3.1358]
80 [0.1712 0.4058 3.167 ]
100 [0.178  0.4123 3.1751]
120 [0.1756 0.4109 3.1716]
140 [0.1745 0.4103 3.1613]
160 [0.1744 0.4098 3.1571]
180 [0.1754 0.4112 3.1617]
200 [0.1772 0.4125 3.1642]
220 [0.1767 0.4124 3.1642]
240 [0.1731 0.407  3.1619]
260 [0.171  0.4048 3.136 ]
280 [0.1689 0.402  3.106 ]
300 [0.1728 0.4057 3.1359]
320 [0.1686 0.4022 3.1193]
340 [0.1712 0.4041 3.1182]
360 [0.1694 0.405  3.1366]
380 [0.1735 0.4088 3.1481]
400 [0.1732 0.4078 3.1605]
420 [0.1735 0.4092 3.1984]
440 [0.1637 0.3998 3.166 ]
460 [0.1627 0.3994 3.1505]
480 [0.1587 0.3942 3.1106]
500 [0.1604 0.3963 3.1086]
520 [0.158  0.3951 3.1105]
540 [0.1564 0.3943 3.096 ]
stats ymax pct [0.38477423 0.39371788 0.40599496 0 …[+22 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];t=T[:,:3,3]
for a,b in [(0,120),(0,204),(204,420),(420,553),(160,450)]:
 print(a,b,'mean',np.round(t[a:b].mean(0),4),'median',np.round(np.median(t[a:b],axis=0),4),'std',np.round(t[a:b].std(0),4))
PY"
```
<details><summary>output</summary>

```
0 120 mean [0.1563 0.2919 3.1157] median [0.1562 0.2898 3.1187] std [0.0034 0.0061 0.0345]
0 204 mean [0.1569 0.295  3.13  ] median [0.1575 0.2979 3.1413] std [0.0028 0.006  0.0322]
204 420 mean [0.1593 0.2964 3.1493] median [0.1584 0.2969 3.1491] std [0.0048 0.0029 0.0273]
420 553 mean [0.1624 0.2845 3.124 ] median [0.1623 0.2838 3.1078] std [0.0027 0.0058 0.0411]
160 450 mean [0.1597 0.2966 3.1527] median [0.1583 0.2971 3.1504] std [0.0048 0.0035 0.0275]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_smooth.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, nvdiffrast.torch as dr, cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
torch.manual_seed(0);dev='cuda';mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev);Fc=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));r=np.load('/work/inputs/reconstruction.npz');T0=r['T_world_mesh'].astype(np.float64)
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][:]
gl=dr.RasterizeCudaContext();S=160
# precompute crops
centers=[]
for i in range(553):
 y,x=np.where(om[i]);centers.append((int(x.mean())-S//2,int(y.mean())-S//2))
def render(T,i):
 x0,y0=centers[i];TT=torch.tensor(T,device=dev,dtype=torch.float32);vv=V@TT[:3,:3].T+TT[:3,3];vv=vv@torch.tensor(Tcw[:3,:3],device=dev).T+torch.tensor(Tcw[:3,3],device=dev)
 z=vv[:,2];k=K.copy();k[0,2]-=x0;k[1,2]-=y0;fx,fy=k[0,0],k[1,1];cx,cy=k[0,2],k[1,2];Hh=Ww=S
 xn=2*(fx*vv[:,0]/z+cx)/Ww-1;yn=2*(fy*vv[:,1]/z+cy)/Hh-1;zn=(z-.1)/(20-.1)*2-1;pos=torch.stack([xn,yn,zn,torch.ones_like(z)],1)[None]
 rast,_=dr.rasterize(gl,pos,Fc,resolution=[Hh,Ww]);m=rast[0,:,:,3]>0
 return m.cpu().numpy()
def evalT(T,frames):
 vals=[]
 for i in frames:
  m=render(T[i],i);gt=om …[+1966 chars]
```
<details><summary>output</summary>

```
static 0 120 base [ 8.82253513e-01  2.27770156e+00 -1.08482394e+00  3.29348333e+03]
const [ 8.76107111e-01  2.21427427e+00 -1.63574925e+00  3.28000000e+03]
static 0 204 base [ 8.90538125e-01  2.16715509e+00 -7.77719237e-01  3.26647549e+03]
const [ 8.97998627e-01  2.24127034e+00 -1.84782482e-01  3.25700000e+03]
static 420 553 base [ 8.81977674e-01  2.01222063e+00 -1.56338367e+00  3.56482707e+03]
const [ 8.86924207e-01  2.08381467e+00 -1.41965316e+00  3.60400000e+03]
static 480 553 base [ 8.84551437e-01  1.90849166e+00 -1.87703031e+00  3.63438356e+03]
const [ 8.94620720e-01  1.82988150e+00 -1.61813343e+00  3.62400000e+03]
base all [ 8.88239941e-01  2.07896546e+00 -9.37680684e-01  3.34975769e+03] early [ 8.90538125e-01  2.16715509e+00 -7.77719237e-01  3.26647549e+03] move [ 8.89925368e-01  2. …[+1796 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static.py <<'PY'
exec(open('/work/scratch/eval_smooth.py').read().split('# static variants')[0])
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

def fit_static(a,b):
 inds=np.arange(a,b);frames=list(range(a,b,max(2,(b-a)//25)))
 Rm=Rot.from_matrix(T0[inds,:3,:3]).mean().as_matrix();tm=np.median(T0[inds,:3,3],axis=0)
 def obj(d):
  Q=T0.copy();Q[a:b,:3,:3]=Rm;Q[a:b,:3,3]=tm+np.asarray(d)
  ev=evalT(Q,frames);return -ev[:,0].mean()
 print('fit',a,b,'start iou',-obj([0,0,0]), 'tm',tm)
 res=minimize(obj,[0,0,0],method='Powell',options={'xtol':1e-4,'ftol':1e-6,'maxiter':100,'disp':True})
 print('res',res.x,'iou',-res.fun)
 Q=T0.copy();Q[a:b,:3,:3]=Rm;Q[a:b,:3,3]=tm+res.x
 return Q,Rm,tm+res.x
Q0,R0,t0=fit_static(0,120);Q1,R1,t1=fit_static(480,553)
np.savez('/work/scratch/static_opt.npz',R0=R0,t0=t0,R1=R1,t1=t1)
print('t0',t0,'t1',t1)
PY
python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
fit 0 120 start iou 0.8763733759826101 tm [0.15615627 0.28976478 3.11874866]
<string>:22: RuntimeWarning: Mean of empty slice.
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Optimization terminated successfully.
         Current function value: -0.959085
         Iterations: 3
         Function evaluations: 239
res [-0.00882762  0.00692175 -0.00015999] iou 0.9590847955499407
fit 480 553 start iou 0.8949070136116709 tm [0.16108833 0.28144488 3.09523439]
Optimization terminated successfully.
         Current function value: -0.959878
         Iterations: 4
         Function evaluations: 306
res [-0.00783557  0.00711899 -0.00039693] iou 0.9598777023268608
t0 [0.14732865 0.29668 …[+54 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();uv=m.visual.uv[m.faces]; area=.5*np.abs(np.cross(uv[:,1]-uv[:,0],uv[:,2]-uv[:,0])); d=np.linalg.norm(np.stack([uv[:,1]-uv[:,0],uv[:,2]-uv[:,1],uv[:,0]-uv[:,2]],1),axis=2)
print('area pct',np.percentile(area,[0,1,5,25,50,75,95,99,100]));print('edge pct',np.percentile(d,[0,25,50,75,90,95,99,100]));
print('faces area >.01',np.sum(area>.01),'triangles',len(area))
# PLY export with texture perhaps and render via open3d? check trimesh save obj
PY"
```
<details><summary>output</summary>

```
area pct [7.61669061e-11 4.45406712e-08 1.68683907e-07 6.42730668e-07
 1.72384626e-06 5.49259861e-06 1.52251421e-05 2.71107264e-05
 1.14485824e-04]
edge pct [3.61631872e-06 1.42371577e-03 2.59565416e-03 4.34759732e-03
 6.12163164e-03 7.26539910e-03 9.90040938e-03 2.16453830e-02]
faces area >.01 0 triangles 98533

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 print('\\n',p)
 x=torch.load(p,map_location='cpu',weights_only=False)
 print(type(x))
 if isinstance(x,dict):
  for k,v in x.items():
   try: print(k,type(v),getattr(v,'shape',None),getattr(v,'dtype',None))
   except: print(k,type(v))
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
<class 'dict'>
schema <class 'str'> None None
frames <class 'list'> None None
frame_meta <class 'list'> None None
kid <class 'int'> None None
checkpoint <class 'dict'> None None
mhr_inference <class 'dict'> None None
config <class 'dict'> None None
mesh_diameter <class 'numpy.ndarray'> (553,) float32
mhr_neutral_height_init <class 'numpy.ndarray'> (553,) float32
mhr_spatial_scale <class 'numpy.ndarray'> (553,) float32
K_rois <class 'numpy.ndarray'> (553, 3, 3) float32
bboxes <class 'numpy.ndarray'> (553, 4) float32
faces <class 'numpy.ndarray'> (36874, 3) int32
pr <class 'dict'> None None
pr_initial <class 'dict'> None None
in <class 'dict'> None None
gt <class 'dict'> None None
observations <class 'dict'> None None
raw <class 'dict'> None None
metadata <c …[+879 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
for sec in ['pr','pr_initial','in','gt','observations','raw','postopt','metadata']:
 print('\\n##',sec, type(x.get(sec)))
 d=x.get(sec)
 if isinstance(d,dict):
  for k,v in d.items(): print(k,type(v),getattr(v,'shape',None),getattr(v,'dtype',None), (v if not hasattr(v,'shape') else ''))
PY"
```
<details><summary>output</summary>

```

## pr <class 'dict'>
pose_abs <class 'numpy.ndarray'> (553, 4, 4) float32 
pose_abs_1st_delta <class 'numpy.ndarray'> (553, 4, 4) float32 
mhr_global_rot6d <class 'numpy.ndarray'> (553, 6) float32 
mhr_trans <class 'numpy.ndarray'> (553, 3) float32 
mhr_body_pose_cont <class 'numpy.ndarray'> (553, 260) float32 
mhr_hand <class 'numpy.ndarray'> (553, 108) float32 
mhr_shape <class 'numpy.ndarray'> (553, 45) float32 
mhr_scale <class 'numpy.ndarray'> (553, 28) float32 
mhr_face <class 'numpy.ndarray'> (553, 72) float32 
contact_logits <class 'numpy.ndarray'> (553, 2) float32 
pose_abs_postopt <class 'numpy.ndarray'> (553, 4, 4) float32 

## pr_initial <class 'dict'>
pose_abs <class 'numpy.ndarray'> (553, 4, 4) float32 
pose_abs_1st_delta <class 'numpy.ndarray'> (553, 4, 4) float32 
mhr_glob …[+31252 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
for k in ['pr','pr_initial','in']:
 print(k,'keys',x[k].keys())
for key in ['pose_abs','pose_abs_postopt']:
 try:
  P=x['pr'][key];print(key,P.shape,P[0],P[250],P[550])
 except Exception as e: print(e)
r=np.load('/work/inputs/reconstruction.npz');print('coco',r['T_cam_mesh_coconet'][250]); print('fp',r['T_cam_mesh_foundationpose'][250])
PY"
```
<details><summary>output</summary>

```
pr keys dict_keys(['pose_abs', 'pose_abs_1st_delta', 'mhr_global_rot6d', 'mhr_trans', 'mhr_body_pose_cont', 'mhr_hand', 'mhr_shape', 'mhr_scale', 'mhr_face', 'contact_logits', 'pose_abs_postopt'])
pr_initial keys dict_keys(['pose_abs', 'pose_abs_1st_delta', 'mhr_global_rot6d', 'mhr_trans', 'mhr_body_pose_cont', 'mhr_hand', 'mhr_shape', 'mhr_scale', 'mhr_face', 'contact_logits'])
in keys dict_keys(['pose_abs', 'mhr_global_rot6d', 'mhr_trans', 'mhr_body_pose_cont', 'mhr_hand', 'mhr_shape', 'mhr_scale', 'mhr_face'])
pose_abs (553, 4, 4) [[-9.92422581e-01 -1.22162007e-01 -1.31847672e-02 -3.58169824e-02]
 [-9.19100057e-05  1.08043402e-01 -9.94146228e-01  1.08105458e-01]
 [ 1.22871436e-01 -9.86611784e-01 -1.07235938e-01  3.18338609e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.000000 …[+1545 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
P=x['pr']['pose_abs'];Q=x['pr']['pose_abs_postopt']
d=Q[:,:3,3]-P[:,:3,3];print('postopt trans delta mean/std/min/max',d.mean(0),d.std(0),np.abs(d).max(0));
for i in range(0,553,25): print(i,np.round(d[i],4))
PY"
```
<details><summary>output</summary>

```
postopt trans delta mean/std/min/max [0. 0. 0.] [0. 0. 0.] [0. 0. 0.]
0 [0. 0. 0.]
25 [0. 0. 0.]
50 [0. 0. 0.]
75 [0. 0. 0.]
100 [0. 0. 0.]
125 [0. 0. 0.]
150 [0. 0. 0.]
175 [0. 0. 0.]
200 [0. 0. 0.]
225 [0. 0. 0.]
250 [0. 0. 0.]
275 [0. 0. 0.]
300 [0. 0. 0.]
325 [0. 0. 0.]
350 [0. 0. 0.]
375 [0. 0. 0.]
400 [0. 0. 0.]
425 [0. 0. 0.]
450 [0. 0. 0.]
475 [0. 0. 0.]
500 [0. 0. 0.]
525 [0. 0. 0.]
550 [0. 0. 0.]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/p3d_render.py <<'PY'
import numpy as np, torch, trimesh, json, cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV, PerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, PointLights, AmbientLights
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer.mesh.textures import TexturesUV
from scipy.spatial.transform import Rotation as Rot
dev='cuda';m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev)[None];F=torch.tensor(m.faces,dtype=torch.int64,device=dev)[None];UV=torch.tensor(m.visual.uv,dtype=torch.float32,device=dev)[None];tex=torch.tensor(np.array(m.visual.material.baseColorTexture),dtype=torch.float32,device=dev)[None]/255
mesh=Meshes(verts=V,faces=F,textures=TexturesUV(maps=tex,faces_uvs=F,verts_uvs=UV))
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc);r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
H,W=1152,1536
Rcw=torch.tensor(Tcw[:3,:3],device=dev)[None];tcw=torch.tensor(Tcw[:3,3],device=dev)[None]
camera=PerspectiveCameras(K=torch.tensor(K,device=dev)[None],R=Rcw,T=tcw,device=dev)
raster=MeshRasterizer(cameras=camera,raster_settings=RasterizationSettings(image_size=(H,W),blur_radius=0,faces_per_pixel=1))
renderer=MeshRenderer(raste …[+542 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/p3d_render.py", line 19, in <module>
    im=renderer(mm)[0].cpu().numpy()
       ^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/mesh/renderer.py", line 63, in forward
    fragments = self.rasterizer(meshes_world, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/t …[+1277 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/p3d_render.py';s=open(p).read();s=s.replace(\"camera=PerspectiveCameras(K=torch.tensor(K,device=dev)[None]\", \"K44=np.eye(4,dtype=np.float32);K44[:3,:3]=K\\ncamera=PerspectiveCameras(K=torch.tensor(K44,device=dev)[None]\");open(p,'w').write(s)
PY
python /work/scratch/p3d_render.py"
```
<details><summary>output</summary>

```
done

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat >> /work/scratch/eval_smooth2.py <<'PY'
PY
cat > /work/scratch/eval_shift.py <<'PY'
exec(open('/work/scratch/eval_smooth.py').read().split('# static variants')[0])
from scipy.optimize import minimize
# constant world shift all frames
frames=list(range(0,553,5))
def obj(d):
 Q=T0.copy();Q[:,:3,3]+=np.asarray(d);return -evalT(Q,frames)[:,0].mean()
print('start',-obj([0,0,0]))
for d in [[-.008,.007,0],[-.008,.007,.01],[-.008,.007,-.01],[-.005,.005,0],[0,0,0]]: print(d,-obj(d))
res=minimize(obj,[-.008,.007,0],method='Powell',options={'xtol':1e-4,'ftol':1e-6,'maxiter':100,'disp':True});print(res.x,-res.fun)
Q=T0.copy();Q[:,:3,3]+=res.x
for name,X in [('base',T0),('shift',Q)]:
 ev=evalT(X,range(553));print(name,ev.mean(0),ev[:204].mean(0),ev[204:420].mean(0),ev[420:].mean(0))
np.save('/work/scratch/global_shift.npy',res.x)
PY
python /work/scratch/eval_shift.py"
```
<details><summary>output</summary>

```
start 0.8890367240537856
[-0.008, 0.007, 0] 0.9251396222091471
[-0.008, 0.007, 0.01] 0.9276808956457736
[-0.008, 0.007, -0.01] 0.9217778477161624
[-0.005, 0.005, 0] 0.9271295581813476
[0, 0, 0] 0.8890367240537856
<string>:22: RuntimeWarning: Mean of empty slice.
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Optimization terminated successfully.
         Current function value: -0.932989
         Iterations: 6
         Function evaluations: 533
[-0.00778672  0.00373527 -0.00585133] 0.9329891226076892
base [ 8.88239941e-01  2.07896546e+00 -9.37680684e-01  3.34975769e+03] [ 8.90538125e-01  2.16715509e+00 -7.77719237e-01  3.26647549e+03] [ 8.89925368e-01  2.03677276e+00 -7.0348 …[+350 chars]
```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][:]
vals=[]
for i in range(553):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read();hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
 # blue hue 85-135 saturation >80 val >50
 bm=(hsv[:,:,0]>85)&(hsv[:,:,0]<140)&(hsv[:,:,1]>70)&(hsv[:,:,2]>50)&om[i]
 y,x=np.where(bm);vals.append([len(x),x.mean() if len(x) else np.nan,y.mean() if len(y) else np.nan,x.min() if len(x) else np.nan,x.max() if len(x) else np.nan])
vals=np.array(vals)
for i in range(0,553,20):print(i,np.round(vals[i],1))
print('max',np.nanmax(vals[:,0]),'pct',np.nanpercentile(vals[:,0],[0,25,50,75,90,95,99,100]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
KeyboardInterrupt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][:]
vals=[]
for i in range(553):
 ok,bgr=cap.read();hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
 bm=(hsv[:,:,0]>85)&(hsv[:,:,0]<140)&(hsv[:,:,1]>70)&(hsv[:,:,2]>50)&om[i]
 y,x=np.where(bm);vals.append([len(x),x.mean() if len(x) else np.nan,y.mean() if len(y) else np.nan,x.min() if len(x) else np.nan,x.max() if len(x) else np.nan])
vals=np.array(vals);np.save('/work/scratch/blue.npy',vals)
for i in range(0,553,20):print(i,np.round(vals[i],1))
print('max',np.nanmax(vals[:,0]),'pct',np.nanpercentile(vals[:,0],[0,25,50,75,90,95,99,100]))
PY"
```
<details><summary>output</summary>

```
0 [ 0. nan nan nan nan]
20 [ 0. nan nan nan nan]
40 [ 0. nan nan nan nan]
60 [ 0. nan nan nan nan]
80 [ 0. nan nan nan nan]
100 [ 0. nan nan nan nan]
120 [ 0. nan nan nan nan]
140 [ 0. nan nan nan nan]
160 [ 0. nan nan nan nan]
180 [ 0. nan nan nan nan]
200 [ 0. nan nan nan nan]
220 [ 0. nan nan nan nan]
240 [ 0. nan nan nan nan]
260 [ 0. nan nan nan nan]
280 [ 45.  780.4 613.5 778.  783. ]
300 [ 41.  780.5 613.7 778.  782. ]
320 [ 29.  779.  614.6 778.  781. ]
340 [268.  779.7 615.7 770.  789. ]
360 [459.  767.8 617.2 752.  783. ]
380 [543.  759.4 617.3 740.  778. ]
400 [316.  745.1 616.5 734.  757. ]
420 [183.  730.6 613.5 724.  737. ]
440 [180.  730.5 613.5 724.  737. ]
460 [176.  730.6 613.2 724.  737. ]
480 [172.  730.6 613.3 724.  737. ]
500 [183.  730.6 613.  724.  737. ]
520 [189.  …[+134 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,cv2
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();uv=m.visual.uv;tex=np.array(m.visual.material.baseColorTexture);h,w=tex.shape[:2]
u=np.clip((uv[:,0]*(w-1)).round().astype(int),0,w-1);v=np.clip(((1-uv[:,1])*(h-1)).round().astype(int),0,h-1);rgb=tex[v,u]
hsv=cv2.cvtColor(rgb.reshape(-1,1,3),cv2.COLOR_RGB2HSV).reshape(-1,3);blue=(hsv[:,0]>85)&(hsv[:,0]<140)&(hsv[:,1]>70)&(hsv[:,2]>50)
print('blue verts',blue.sum(),len(blue));print('uv blue bounds',uv[blue].min(0),uv[blue].max(0));print('colors',np.unique(rgb[blue],axis=0)[:20])
# face any/all blue
fb=blue[m.faces];print('faces any/all',fb.any(1).sum(),fb.all(1).sum())
# save image highlight actual points? texture visual
im=tex.copy();im[v[blue],u[blue]]=[0,255,0];cv2.imwrite('/work/scratch/blueverts.png',cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
np.save('/work/scratch/blueverts.npy',blue)
PY"
```
<details><summary>output</summary>

```
blue verts 16023 295487
uv blue bounds [0.002088   0.00331801] [0.99704599 0.997796  ]
colors [[  0  33  84]
 [  0  34  76]
 [  0  37  82]
 [  0  37  88]
 [  0  37  89]
 [  0  39  90]
 [  0  40  92]
 [  0  41  93]
 [  0  43  98]
 [  0  44  91]
 [  0  44 101]
 [  0  47 106]
 [  0  48  96]
 [  0  49 107]
 [  0  50  99]
 [  0  50 103]
 [  0  51 108]
 [  1  35  83]
 [  1  38  83]
 [  1  38  90]]
faces any/all 5921 4781

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_blue.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, nvdiffrast.torch as dr
torch.manual_seed(0);dev='cuda';S=180
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);Fc=torch.tensor(m.faces,dtype=torch.int32,device=dev);blue=torch.tensor(np.load('/work/scratch/blueverts.npy').astype(np.float32),device=dev)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][:]
obs=np.load('/work/scratch/blue.npy');gl=dr.RasterizeCudaContext()
def render(T,i):
 y,x=np.where(om[i]);x0,y0=int(x.mean())-S//2,int(y.mean())-S//2
 TT=torch.tensor(T,device=dev,dtype=torch.float32);vv=V@TT[:3,:3].T+TT[:3,3];vv=vv@torch.tensor(Tcw[:3,:3],device=dev).T+torch.tensor(Tcw[:3,3],device=dev)
 z=vv[:,2];k=K.copy();k[0,2]-=x0;k[1,2]-=y0;H=W=S;fx,fy=k[0,0],k[1,1];cx,cy=k[0,2],k[1,2]
 xn=2*(fx*vv[:,0]/z+cx)/W-1;yn=2*(fy*vv[:,1]/z+cy)/H-1;zn=(z-.1)/(20-.1)*2-1;pos=torch.stack([xn,yn,zn,torch.ones_like(z)],1)[None]
 rast,_=dr.rasterize(gl,pos,Fc,resolution=[H,W]);bm,_=dr.interpolate(blue[None,:,None],rast,Fc);bm=bm[0,:,:,0]
 return (bm>0.5).cpu().numpy(),(rast[0,:,:,3]>0).cpu().numpy(),x0,y0
res=[]
for i in range(200,553):
 bm,mm,x0,y0=render(T,i);y,x=np.where( …[+632 chars]
```
<details><summary>output</summary>

```
/work/scratch/eval_blue.py:9: UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3683.)
  TT=torch.tensor(T,device=dev,dtype=torch.float32);vv=V@TT[:3,:3].T+TT[:3,3];vv=vv@torch.tensor(Tcw[:3,:3],device=dev).T+torch.tensor(Tcw[:3,3],device=dev)
Traceback (most recent call last):
  File "/work/scratch/eval_blue.py", line 16, in <module>
    bm,mm,x0,y0=render(T,i);y,x=np.where(bm)
                ^^^^^^^^^^^
  File "/work/scratch/eval_blue.py", line 9, in render
    TT=torch.tensor(T,device=d …[+313 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/bm,mm,x0,y0=render(T,i)/bm,mm,x0,y0=render(T[i],i)/' /work/scratch/eval_blue.py
python /work/scratch/eval_blue.py"
```
<details><summary>output</summary>

```
200 obs [ 0. nan nan nan nan] pred [200.   0.  nan  nan  nan  nan]
220 obs [ 0. nan nan nan nan] pred [220.   0.  nan  nan  nan  nan]
240 obs [ 0. nan nan nan nan] pred [240.   0.  nan  nan  nan  nan]
260 obs [ 0. nan nan nan nan] pred [260.   0.  nan  nan  nan  nan]
280 obs [ 45.  780.4 613.5 778.  783. ] pred [280.   85.  783.3 614.3 780.  786. ]
300 obs [ 41.  780.5 613.7 778.  782. ] pred [300.   87.  782.9 614.6 780.  786. ]
320 obs [ 29.  779.  614.6 778.  781. ] pred [320.  106.  782.1 613.8 779.  786. ]
340 obs [268.  779.7 615.7 770.  789. ] pred [340.  296.  781.5 615.9 772.  791. ]
360 obs [459.  767.8 617.2 752.  783. ] pred [360.  507.  768.6 616.8 752.  784. ]
380 obs [543.  759.4 617.3 740.  778. ] pred [380.  612.  760.8 617.3 740.  780. ]
400 obs [316.  745.1 616.5 734.  7 …[+716 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_blue.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, nvdiffrast.torch as dr, cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
torch.manual_seed(0);dev='cuda';S=180
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);Fc=torch.tensor(m.faces,dtype=torch.int32,device=dev);blue=torch.tensor(np.load('/work/scratch/blueverts.npy').astype(np.float32),device=dev);blue=blue[None,:,None]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));T0=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][:]
# load blue image masks sequentially into crops
cap=cv2.VideoCapture('/work/inputs/video.mp4'); obm=np.zeros((553,S,S),bool); crop0=np.zeros((553,2),int)
for i in range(553):
 ok,bgr=cap.read();hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);bm=(hsv[:,:,0]>85)&(hsv[:,:,0]<140)&(hsv[:,:,1]>70)&(hsv[:,:,2]>50)&om[i]
 y,x=np.where(om[i]);x0,y0=int(x.mean())-S//2,int(y.mean())-S//2;crop0[i]=[x0,y0];obm[i]=bm[y0:y0+S,x0:x0+S]
gl=dr.RasterizeCudaContext()
def render(T,i):
 x0,y0=crop0[i];TT=torch.tensor(T,device=dev,dtype=torch.float32);vv=V@TT[:3,:3].T+TT[:3,3];vv=vv@torch.tensor(Tcw[:3,:3],device=dev).T+torch.tensor(Tcw[:3,3],device=dev)
 z=vv[:,2];k …[+1379 chars]
```
<details><summary>output</summary>

```
base
[0, 0, 0, 0] -0.6973714924067542 0.6515821023820374 dynamic 0.622228141356337 static 0.6962001231411024
[-0.17453292519943295, 0, 0, 0] -0.5632298757626587 0.5211188604193613 dynamic 0.4902775894916475 static 0.5679975922294862
[-0.08726646259971647, 0, 0, 0] -0.6319225672117852 0.5883424503439221 dynamic 0.5588001354264116 static 0.6332467690185386
[-0.03490658503988659, 0, 0, 0] -0.6691364904950673 0.6241104218166827 dynamic 0.5950605718608916 static 0.668266193749485
[0.0, 0, 0, 0] -0.6973714924067542 0.6515821023820374 dynamic 0.622228141356337 static 0.6962001231411024
[0.03490658503988659, 0, 0, 0] -0.7198717268834162 0.6735543295528399 dynamic 0.6432357559713419 static 0.7196385613967167
[0.08726646259971647, 0, 0, 0] -0.7395316499686251 0.6930130531489558 dynamic 0.65646709839 …[+508 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_cand.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz');T0=r['T_world_mesh'].astype(np.float64)
a0,a1=204,420
# mean orientation static segments
def static_pose(T,a,b):
 Rm=Rot.from_matrix(T[a:b,:3,:3]).mean().as_matrix();tm=np.median(T[a:b,:3,3],axis=0);return Rm,tm
R0,t0=static_pose(T0,0,a0);R1,t1=static_pose(T0,a1,553)
print('static t0',t0,'t1',t1)
# yaw trajectory baseline
def yaw_of(R): return np.unwrap(np.arctan2(R[:,2,0],R[:,0,0]))
yaw=yaw_of(T0[:,:3,:3]); yaws=savgol_filter(yaw,15,2)
# Apply candidate helpers
for gname,g in [('none',np.zeros(3)),('g',np.array([-.008,.004,-.006])),('g2',np.array([-.008,.007,-.003]))]:
 T=T0.copy();
 # smooth dynamic rotation + translation
 for i in range(a0,a1):
  dy=yaws[i]-yaw[i];T[i,:3,:3]=Rot.from_rotvec([0,dy,0]).as_matrix()@T[i,:3,:3]
 for j in range(3): T[a0:a1,j,3]=savgol_filter(T0[a0:a1,j,3],15,2)
 # static exact
 T[:a0,:3,:3]=R0;T[:a0,:3,3]=t0+g
 T[a1:,:3,:3]=R1;T[a1:,:3,3]=t1+g
 # blend translation correction boundary 190:215 409:430? and set T to static already; dynamic t add g? smooth g globally
 T[a0:a1,:3,3]+=g
 np.savez(f'/work/scratch/cand_{gname}.npz',T=T,R0=R0,t0=t0+g,R1=R1,t1=t1+g)
 # stats
 print(gname,'bottom y static',end=' ')
 import trimesh
 m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=m.vertices
 for i in [0,5 …[+168 chars]
```
<details><summary>output</summary>

```
static t0 [0.15748668 0.29786712 3.14133894] t1 [0.16232444 0.28375372 3.10783148]
none bottom y static 0 0.4096;50 0.4096;100 0.4096;150 0.4096;200 0.4096;250 0.4093;300 0.4057;350 0.404;400 0.4088;450 0.3966;500 0.3966;550 0.3966;
g bottom y static 0 0.4136;50 0.4136;100 0.4136;150 0.4136;200 0.4136;250 0.4133;300 0.4097;350 0.408;400 0.4128;450 0.4006;500 0.4006;550 0.4006;
g2 bottom y static 0 0.4166;50 0.4166;100 0.4166;150 0.4166;200 0.4166;250 0.4163;300 0.4127;350 0.411;400 0.4158;450 0.4036;500 0.4036;550 0.4036;

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p) as f:
 print('attrs',dict(f.attrs))
 def vis(n,o):
  if isinstance(o,h5py.Dataset): print(n,o.shape,o.dtype,dict(o.attrs))
 f.visititems(vis)
PY"
```
<details><summary>output</summary>

```
attrs {'complete': True, 'depth_alignment_input_identity_json': '{"depth_backend":"moge2","depth_model_id":"Ruicheng/moge-2-vitl-normal","depth_model_revision":"b135031bae30b5ac2ae141a0e68717795ce38340","depth_source_commit":"925b8ed835a7a9cdb7578ba15c658a0afc969030","direct_mhr":{"mtime_ns":1791233135420693924,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000020/mhr_init/episode_000020.pkl","size":2468619},"export":{"mtime_ns":1791233016177694379,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000020/export/episode_000020/wild_export.json","size":2798},"raw_depth":{"mtime_ns":1791232955327694611,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000020/depth/moge2/raw.npy","size":1957036160},"raw_depth_report":{"mtime_ns":1791232957593694602,"path":"/home/hanzhou/Projects/v2d/run/ …[+559 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,io,cv2
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p) as f:
 for key in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left','frame_names/front_stereo_camera_left']:
  x=f[key][0];print(key,type(x),getattr(x,'shape',None),getattr(x,'dtype',None));
  if isinstance(x,(bytes,np.bytes_)): print(x[:100])
  elif hasattr(x,'shape') and x.shape==(): 
   try: print('scalar',x[()])
   except Exception as e: print(e)
 print('scale shift',f['alignment/front_stereo_camera_left/scale'][:10],f['alignment/front_stereo_camera_left/shift'][:10])
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left <class 'numpy.ndarray'> (649966,) uint8
raw/front_stereo_camera_left <class 'numpy.ndarray'> (596861,) uint8
frame_names/front_stereo_camera_left <class 'bytes'> None None
b'000000'
scale shift [1.1150279 1.1032774 1.104247  1.0984145 1.0923247 1.097358  1.0967562
 1.088642  1.1004287 1.0982538] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for k in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  b=f[k][250].tobytes();im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED);print(k,im.shape,im.dtype,im.min(),im.max(),np.percentile(im,[1,50,99]))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (1152, 1536) uint16 1795 10424 [1877. 4178. 9496.]
raw/front_stereo_camera_left (1152, 1536) uint16 1579 9169 [1651. 3675. 8353.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for i in [0,250,552]:
  b=f['aligned/front_stereo_camera_left'][i].tobytes();d=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED)
  print('frame',i)
  for y,x in [(609,756),(640,756),(660,756),(700,756),(609,700),(609,810),(500,756),(609,600),(900,756)]:print((x,y),d[y,x])
PY"
```
<details><summary>output</summary>

```
frame 0
(756, 609) 3166
(756, 640) 3200
(756, 660) 2922
(756, 700) 3282
(700, 609) 7426
(810, 609) 7854
(756, 500) 9349
(600, 609) 2853
(756, 900) 2925
frame 250
(756, 609) 3078
(756, 640) 3099
(756, 660) 2830
(756, 700) 3303
(700, 609) 4125
(810, 609) 7932
(756, 500) 9566
(600, 609) 6448
(756, 900) 2951
frame 552
(756, 609) 3243
(756, 640) 3255
(756, 660) 2947
(756, 700) 3392
(700, 609) 8227
(810, 609) 8113
(756, 500) 9699
(600, 609) 2158
(756, 900) 2986

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2,json,h5py,open3d as o3d
from scipy.spatial import cKDTree
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 b=f['aligned/front_stereo_camera_left'][0].tobytes();d=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][0];pm=f['person'][0]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam'])
y,x=np.mgrid[0:1152,0:1536];Z=d;X=(x-K[0,2])/K[0,0]*Z;Y=(y-K[1,2])/K[1,1]*Z
P=np.stack([X,Y,Z],-1)@Twc[:3,:3].T+Twc[:3,3]
roi=(x>350)&(x<1200)&(y>620)&(y<900)&(~om)&(~pm)&(Z>2)&(Z<5)
pts=P[roi];print('pts',len(pts),np.percentile(pts,[0,5,50,95,100],axis=0))
# open3d RANSAC
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts));pc=pc.voxel_down_sample(.01)
for it in [1000]:
 model,inl=pc.segment_plane(distance_threshold=.01,ransac_n=3,num_iterations=it);print('model',model,'inliers',len(inl),len(pc.points))
 a,b,c,dd=model;n=np.array([a,b,c]);print('norm',n,'dist object center',n@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0,:3,3]+dd)
PY"
```
<details><summary>output</summary>

```
pts 143813 [[-2.2310818   0.31113765  2.83514979]
 [-1.57601963  0.46756898  2.99145767]
 [ 0.37085111  1.17407173  3.64119374]
 [ 2.12670768  1.25200226  4.86194032]
 [ 2.85087201  1.27923983  5.17863427]]
model [ 0.01971612  0.99609083  0.08610654 -1.50886654] inliers 56003 68240
norm [0.01971612 0.99609083 0.08610654] dist object center -0.9535310318034215

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2,json,open3d as o3d
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:d=cv2.imdecode(np.frombuffer(f['aligned/front_stereo_camera_left'][0].tobytes(),np.uint8),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][0];pm=f['person'][0]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam']);y,x=np.mgrid[:1152,:1536];Z=d;X=(x-K[0,2])/K[0,0]*Z;Y=(y-K[1,2])/K[1,1]*Z;P=np.stack([X,Y,Z],-1)@Twc[:3,:3].T+Twc[:3,3]
roi=(((x-755)/390)**2+((y-690)/120)**2<1)&(~om)&(~pm)&(Z>2)&(Z<4)
pts=P[roi];print('n',len(pts),np.percentile(pts,[0,5,50,95,100],axis=0))
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts));pc=pc.voxel_down_sample(.008)
model,inl=pc.segment_plane(.008,3,3000);print(model,len(inl),len(pc.points));n=np.array(model[:3]);print('dist f0',n@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0,:3,3]+model[3])
PY"
```
<details><summary>output</summary>

```
n 14743 [[-1.20411201  0.14665981  2.87914251]
 [-0.45878029  0.34927915  2.97823213]
 [ 0.13204961  0.72721083  3.470002  ]
 [ 0.81287938  1.17818404  4.00445369]
 [ 1.26703204  1.19133831  4.11476281]]
[ 0.01131551  0.9960516   0.08805204 -0.65405772] 2865 9494
dist f0 -0.093999916621478

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2,json
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:d=cv2.imdecode(np.frombuffer(f['aligned/front_stereo_camera_left'][0].tobytes(),np.uint8),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][0];pm=f['person'][0]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam']);y,x=np.mgrid[:1152,:1536];Z=d;X=(x-K[0,2])/K[0,0]*Z;Y=(y-K[1,2])/K[1,1]*Z;P=np.stack([X,Y,Z],-1)@Twc[:3,:3].T+Twc[:3,3]
for yy in range(580,800,20):
 vals=[]
 for xx in range(400,1101,50):
  if not(om[yy,xx] or pm[yy,xx]): vals.append(P[yy,xx,1])
 print(yy,np.round(vals,2))
PY"
```
<details><summary>output</summary>

```
580 [0.2  0.2  0.26 0.26 0.34 0.3  0.29 0.26 0.21 0.2  0.18]
600 [0.32 0.33 0.44 0.44 0.58 0.51 0.43 0.45 0.34 0.32 0.3 ]
620 [0.44 0.46 0.62 0.62 0.5  0.71 0.7  0.61 0.61 0.45 0.42]
640 [0.57 0.59 0.8  0.37 0.37 0.37 0.79 0.78 0.81 0.7  0.79]
660 [0.69 0.72 0.63 0.93 0.45 0.41 0.4  0.91 0.9  0.9  0.9  0.86 0.87]
680 [0.82 0.85 0.64 0.99 0.99 0.58 0.97 0.97 0.97 0.96 0.95 0.95 0.95]
700 [0.95 1.03 0.74 1.04 1.04 0.72 1.02 1.02 1.02 1.01 1.   1.   1.  ]
720 [1.08 1.08 1.08 1.08 1.07 0.8  1.06 1.06 1.06 1.05 1.05 1.04 1.04]
740 [1.12 1.11 1.12 1.11 1.11 0.93 1.1  1.1  1.09 1.09 1.08 1.08 1.08]
760 [1.15 1.14 1.14 0.97 1.14 1.13 1.08 1.13 1.12 1.12 1.12 1.11 1.11 1.11]
780 [1.17 1.17 1.16 1.16 1.16 1.16 1.13 1.15 1.15 1.15 1.15 1.14 1.14 1.13]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/yaw_from_blue.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
torch.manual_seed(0);dev='cuda';S=180
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);Fc=torch.tensor(m.faces,dtype=torch.int32,device=dev);blue=torch.tensor(np.load('/work/scratch/blueverts.npy').astype(np.float32),device=dev)[None,:,None]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));T0=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64);obs=np.load('/work/scratch/blue.npy')
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][:]
gl=dr.RasterizeCudaContext()
def render(T,i):
 y,x=np.where(om[i]);x0,y0=int(x.mean())-S//2,int(y.mean())-S//2;TT=torch.tensor(T,device=dev,dtype=torch.float32);vv=V@TT[:3,:3].T+TT[:3,3];vv=vv@torch.tensor(Tcw[:3,:3],device=dev).T+torch.tensor(Tcw[:3,3],device=dev);z=vv[:,2]
 k=K.copy();k[0,2]-=x0;k[1,2]-=y0;H=W=S;fx,fy=k[0,0],k[1,1];cx,cy=k[0,2],k[1,2]
 xn=2*(fx*vv[:,0]/z+cx)/W-1;yn=2*(fy*vv[:,1]/z+cy)/H-1;zn=(z-.1)/(20-.1)*2-1;pos=torch.stack([xn,yn,zn,torch.ones_like(z)],1)[None]
 rast,_=dr.rasterize(gl,pos,Fc,resolution=[H,W]);bm,_=dr.interpolate(blue,rast,Fc);bm=(bm[0,:,:,0]>.5).cpu().numpy();y,x=np.whe …[+1441 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/yaw_from_blue.py", line 27, in <module>
    ox,oy,oa=obs[i]
    ^^^^^^^^
ValueError: too many values to unpack (expected 3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/ox,oy,oa=obs\\[i\\]/ox,oy,oa=obs[i,:3]/' /work/scratch/yaw_from_blue.py
python /work/scratch/yaw_from_blue.py"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py:1384: RuntimeWarning: All-NaN slice encountered
  return _nanquantile_unchecked(
corr pct [nan nan nan nan nan nan nan]
260 obs [ 0. nan nan nan nan] predx nan corrdeg nan
270 obs [ 0. nan nan nan nan] predx nan corrdeg nan
280 obs [ 45.  780.4 613.5 778.  783. ] predx 780.0 corrdeg nan
290 obs [ 24.  779.8 613.8 778.  781. ] predx 781.0 corrdeg nan
300 obs [ 41.  780.5 613.7 778.  782. ] predx nan corrdeg nan
310 obs [ 43.  780.4 613.6 778.  782. ] predx nan corrdeg nan
320 obs [ 29.  779.  614.6 778.  781. ] predx 780.0 corrdeg nan
330 obs [185.  780.5 615.  774.  787. ] predx 780.2 corrdeg nan
340 obs [268.  779.7 615.7 770.  789. ] predx 780.0 corrdeg nan
350 obs [371.  772.6 617.3 760.  785. ] predx 780.3 corrdeg nan
36 …[+1768 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/yaw_from_blue.py').read().split('# candidate')[0]
exec(s)
import numpy as np
from scipy.spatial.transform import Rotation as Rot
g=np.array([-.008,.004,-.006])
for i in [300,340,350,360,380,420]:
 for gg in [0,1]:
  TT=T0[i].copy()
  if gg:TT[:3,3]+=g
  print(i,gg,render(TT,i))
PY"
```
<details><summary>output</summary>

```
300 0 [782.8505747126437, 614.5862068965517, 87]
300 1 [780.8977272727273, 615.6818181818182, 88]
340 0 [781.5168918918919, 615.8614864864865, 296]
340 1 [779.4782608695652, 616.9665551839465, 299]
350 0 [773.2632850241546, 615.8140096618357, 414]
350 1 [771.2469733656175, 617.1162227602906, 413]
360 0 [768.5700197238659, 616.7790927021696, 507]
360 1 [766.4518664047151, 618.1434184675835, 509]
380 0 [760.7532679738562, 617.2794117647059, 612]
380 1 [758.7260940032415, 618.4959481361426, 617]
420 0 [732.568720379147, 613.3175355450237, 211]
420 1 [730.5370370370371, 614.6342592592592, 216]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/yaw_from_blue.py').read().split('# candidate')[0];exec(s)
from scipy.spatial.transform import Rotation as Rot
for i in [300,340,350,360,380,420]:
 print('frame',i)
 for deg in [-10,-5,-2,-1,0,1,2,5,10]:
  TT=T0[i].copy();RR=Rot.from_rotvec([0,np.deg2rad(deg),0]).as_matrix();c=TT[:3,3].copy();TT[:3,:3]=RR@TT[:3,:3];print(deg,render(TT,i))
PY"
```
<details><summary>output</summary>

```
frame 300
-10 [782.2222222222222, 612.0, 9]
-5 [782.4583333333334, 614.0416666666666, 48]
-2 [782.8636363636364, 614.3484848484849, 66]
-1 [782.972602739726, 614.1917808219177, 73]
0 [782.8505747126437, 614.5862068965517, 87]
1 [782.9782608695652, 614.2826086956521, 92]
2 [783.0, 614.5102040816327, 98]
5 [782.9268292682926, 614.7560975609756, 123]
10 [782.9811320754717, 615.0377358490566, 159]
frame 340
-10 [783.1059322033898, 615.3220338983051, 236]
-5 [782.2277992277992, 615.5444015444016, 259]
-2 [781.9157894736842, 615.7684210526315, 285]
-1 [781.6621160409557, 615.8054607508533, 293]
0 [781.5168918918919, 615.8614864864865, 296]
1 [781.4191419141914, 615.9438943894389, 303]
2 [781.1414790996785, 615.8971061093248, 311]
5 [780.3516819571865, 615.9571865443424, 327]
10 [779.290055248618 …[+1727 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static_blue.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, nvdiffrast.torch as dr, cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot, Slerp
torch.manual_seed(0);dev='cuda';S=180
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);Fc=torch.tensor(m.faces,dtype=torch.int32,device=dev);battr=torch.tensor(np.load('/work/scratch/blueverts.npy').astype(np.float32),device=dev)[None,:,None]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));T0=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');obm=np.zeros((553,S,S),bool);crop0=np.zeros((553,2),int)
for i in range(553):
 ok,bgr=cap.read();hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);bm=(hsv[:,:,0]>85)&(hsv[:,:,0]<140)&(hsv[:,:,1]>70)&(hsv[:,:,2]>50)&om[i]
 y,x=np.where(om[i]);x0,y0=int(x.mean())-S//2,int(y.mean())-S//2;crop0[i]=[x0,y0];obm[i]=bm[y0:y0+S,x0:x0+S]
gl=dr.RasterizeCudaContext()
def render(T,i):
 x0,y0=crop0[i];TT=torch.tensor(T,device=dev,dtype=torch.float32);vv=V@TT[:3,:3].T+TT[:3,3];vv=vv@torch.tensor(Tcw[:3,:3],device=dev).T+torch.tensor(Tcw[:3,3],device=dev);z=vv[:,2]
 k=K.copy();k[0,2]-=x0;k[1,2]-=y0;H=W=S;fx,fy=k …[+1321 chars]
```
<details><summary>output</summary>

```
start 480 553 -0.75530519986821
Optimization terminated successfully.
         Current function value: -0.889086
         Iterations: 6
         Function evaluations: 761
opt [ 0.05710977 -0.00249896  0.00268064  0.00274359] deg 3.272148710858724 val -0.8890856639449524

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_candidates_blue.py <<'PY'
exec(open('/work/scratch/opt_static_blue.py').read().split('frames=list(range(480,553,2))')[0])
from scipy.signal import savgol_filter
# yaw derivative smooth and static mean rotations
a0,a1=204,420
R0m=Rot.from_matrix(T0[:a0,:3,:3]).mean().as_matrix();t0m=np.median(T0[:a0,:3,3],axis=0)
R1m=Rot.from_matrix(T0[a1:,:3,:3]).mean().as_matrix();t1m=np.median(T0[a1:,:3,3],axis=0)
yaw=np.unwrap(np.arctan2(T0[:,2,0],T0[:,0,0]));ys=savgol_filter(yaw,15,2)
dynframes=list(range(280,470,4));lateframes=list(range(480,553,3))
def make(yawoff,g,static=True,win=15):
 Q=T0.copy()
 for i in range(a0,a1): Q[i,:3,:3]=Rot.from_rotvec([0,ys[i]-yaw[i]+yawoff,0]).as_matrix()@T0[i,:3,:3]
 for j in range(3):Q[a0:a1,j,3]=savgol_filter(T0[a0:a1,j,3],win,2)
 Q[a0:a1,:3,3]+=g
 if static:
  Q[:a0,:3,:3]=Rot.from_rotvec([0,yawoff,0]).as_matrix()@R0m;Q[:a0,:3,3]=t0m+g
  Q[a1:,:3,:3]=Rot.from_rotvec([0,yawoff,0]).as_matrix()@R1m;Q[a1:,:3,3]=t1m+g
 return Q
def ev(Q,frames):
 vals=[]
 for i in frames:
  mm,bm=render(Q[i],i);gt=om[i][crop0[i][1]:crop0[i][1]+S,crop0[i][0]:crop0[i][0]+S];vals.append([iou(mm,gt),iou(bm,obm[i]),bm.sum(),obm[i].sum()])
 return np.array(vals)
# load opt static
z=np.load('/work/scratch/static_blue_opt.npz');yawopt=z['p'][0];gopt=z['p'][1:]
print('opt',np.degrees(yawopt),gopt)
for yawdeg in [0,1,2,3,3.27,4,5]:
 for gn,g in [('z',np.array([-.008,.004,-.006])),('b',gopt),('z2',np.array([-.004,.004,.003]))]:
  Q=make(np. …[+192 chars]
```
<details><summary>output</summary>

```
opt 3.272148710858724 [-0.00249896  0.00268064  0.00274359]
yaw 0 z late [  0.953   0.734 231.    183.2  ] dyn [  0.924   0.675 294.729 239.875]
yaw 0 b late [  0.923   0.79  219.    183.2  ] dyn [  0.907   0.668 290.312 239.875]
yaw 0 z2 late [  0.942   0.789 225.    183.2  ] dyn [  0.918   0.678 292.083 239.875]
yaw 1 z late [  0.955   0.714 221.    183.2  ] dyn [  0.923   0.66  294.958 239.875]
yaw 1 b late [  0.924   0.8   214.    183.2  ] dyn [  0.907   0.677 291.688 239.875]
yaw 1 z2 late [  0.942   0.825 215.    183.2  ] dyn [  0.917   0.684 292.062 239.875]
yaw 2 z late [  0.955   0.728 216.    183.2  ] dyn [  0.922   0.652 295.812 239.875]
yaw 2 b late [  0.923   0.808 212.    183.2  ] dyn [  0.906   0.678 293.625 239.875]
yaw 2 z2 late [  0.943   0.833 213.    183.2  ] dyn [  0.9 …[+1061 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_final_cand.py <<'PY'
import numpy as np, trimesh, json, h5py
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz');T0=r['T_world_mesh'].astype(np.float64);F=len(T0)
a0,a1=204,420;g=np.array([-.008,.007,0.])
def sp(T,a,b):return Rot.from_matrix(T[a:b,:3,:3]).mean().as_matrix(),np.median(T[a:b,:3,3],axis=0)
R0,t0=sp(T0,0,a0);R1,t1=sp(T0,a1,F)
# baseline yaw and smooth
yaw=np.unwrap(np.arctan2(T0[:,2,0],T0[:,0,0])); ys=savgol_filter(yaw,15,2)
Q=T0.copy()
# dynamic
for i in range(a0,a1):
 dy=ys[i]-yaw[i];Q[i,:3,:3]=Rot.from_rotvec([0,dy,0]).as_matrix()@T0[i,:3,:3]
for j in range(3):Q[a0:a1,j,3]=savgol_filter(T0[a0:a1,j,3],15,2)+g[j]
# static
for i in range(0,a0): Q[i,:3,:3]=R0;Q[i,:3,3]=t0+g
for i in range(a1,F): Q[i,:3,:3]=R1;Q[i,:3,3]=t1+g
# blend transitions 8 frames using slerp rotations and lerp trans
for a,b,dir in [(a0,a0+8,1),(a1-8,a1,-1)]:
 pass
# Explicit taper: dynamic curve pulled toward neighboring static at endpoints over 8 frames
for k in range(8):
 s=0.5*(1-np.cos(np.pi*(k+1)/9)) # 0->1
 # at start static->dynamic
 i=a0+k; Rd=Rot.from_matrix(Q[i,:3,:3]);Rs=Rot.from_matrix(R0);Rq=Rot.from_matrix((Rs.inv()*Rd).as_matrix()*s).as_matrix() if False else None
 # slerp small angles using rotation vector log
 rel=(Rot.from_matrix(R0).inv()*Rot.from_matrix(Q[i,:3,:3])).as_rotvec();Q[i,:3,:3]=Rot.from_rotvec(rel*s).as_matrix()@R0
 Q[i,:3,3]=(1-s)*(t0 …[+862 chars]
```
<details><summary>output</summary>

```
base tstep [0.0075075  0.04277331 0.08549317 0.09327658] rotstep [0.70995259 2.21627573 4.07153174 4.99060623] taccsum 0.02693457852815524 racc 14.819968326454946 boundjumps [0.00319724 0.00674651 0.00028891 0.00284509] [0.27639182 0.65135197 0.36022872 0.26417482]
states [0.14948668 0.30486712 3.14133894] [0.14947571 0.30499861 3.14146096] [0.15984949 0.30358638 3.19047783] [0.15432444 0.29075372 3.10783148]
final tstep [0.         0.00501919 0.01395641 0.08381899] rotstep [ 0.          2.58849078  5.78033307 23.37186706] taccsum 0.0017478345873160356 racc 37.234203734664916 boundjumps [0.         0.00017972 0.00054514 0.00088712] [0.         0.02142277 0.11519358 0.09202971]
states [0.14948668 0.30486712 3.14133894] [0.14947571 0.30499861 3.14146096] [0.15984949 0.30358638 3.19047783] [0 …[+33 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];yaw=np.unwrap(np.arctan2(T[:,2,0],T[:,0,0]));yaw=np.degrees(yaw-yaw[0])
for i in range(390,481,2):print(i,np.round(T[i,:3,3],4),round(yaw[i],2))
PY"
```
<details><summary>output</summary>

```
[0.1631 0.2865 3.139 ] -308.88
466 [0.1638 0.2868 3.1571] -308.54
468 [0.1631 0.2869 3.167 ] -307.76
470 [0.1637 0.2885 3.163 ] -308.03
472 [0.163  0.2871 3.1607] -307.91
474 [0.1641 0.2878 3.1584] -308.31
476 [0.1631 0.2854 3.1453] -308.22
478 [0.165  0.2869 3.1565] -308.14
480 [0.1593 0.2823 3.1051] -308.47

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/build_final_cand.py';s=open(p).read().replace('a0,a1=204,420','a0,a1=204,480');open(p,'w').write(s)
PY
python /work/scratch/build_final_cand.py"
```
<details><summary>output</summary>

```
base tstep [0.0075075  0.04277331 0.08549317 0.09327658] rotstep [0.70995259 2.21627573 4.07153174 4.99060623] taccsum 0.02693457852815524 racc 14.819968326454946 boundjumps [0.00319724 0.00674651 0.00028891 0.00284509] [0.27639182 0.65135197 0.36022872 0.26417482]
states [0.14948668 0.30486712 3.14133894] [0.14947571 0.30499861 3.14146096] [0.15952689 0.30432439 3.19252397] [0.15968765 0.30426541 3.19355781]
final tstep [0.00013991 0.00606393 0.01333984 0.06248464] rotstep [0.05672624 2.52281636 5.22412959 7.11049209] taccsum 0.0019011923262748586 racc 24.50108666152435 boundjumps [0.         0.00017972 0.00054514 0.00088712] [0.         0.02142277 0.11519358 0.09202971]
states [0.14948668 0.30486712 3.14133894] [0.14947571 0.30499861 3.14146096] [0.15952689 0.30432439 3.19252397] [0.1596 …[+28 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_windows.py <<'PY'
exec(open('/work/scratch/opt_static_blue.py').read().split('frames=list(range(480,553,2))')[0])
from scipy.signal import savgol_filter
a0,a1=204,480;g=np.array([-.008,.007,0.]);yaw=np.unwrap(np.arctan2(T0[:,2,0],T0[:,0,0]))
R0=Rot.from_matrix(T0[:a0,:3,:3]).mean().as_matrix();t0=np.median(T0[:a0,:3,3],axis=0)+g
R1=Rot.from_matrix(T0[a1:,:3,:3]).mean().as_matrix();t1=np.median(T0[a1:,:3,3],axis=0)+g
def make(win):
 Q=T0.copy();ys=savgol_filter(yaw,win,2)
 for i in range(a0,a1):Q[i,:3,:3]=Rot.from_rotvec([0,ys[i]-yaw[i],0]).as_matrix()@T0[i,:3,:3]
 for j in range(3):Q[a0:a1,j,3]=savgol_filter(T0[a0:a1,j,3],win,2)+g[j]
 Q[:a0,:3,:3]=R0;Q[:a0,:3,3]=t0;Q[a1:,:3,:3]=R1;Q[a1:,:3,3]=t1
 for k in range(8):
  s=.5*(1-np.cos(np.pi*(k+1)/9));i=a0+k;rel=(Rot.from_matrix(R0).inv()*Rot.from_matrix(Q[i,:3,:3])).as_rotvec();Q[i,:3,:3]=Rot.from_rotvec(rel*s).as_matrix()@R0;Q[i,:3,3]=(1-s)*t0+s*Q[i,:3,3]
  i=a1-1-k;rel=(Rot.from_matrix(Q[i,:3,:3]).inv()*Rot.from_matrix(R1)).as_rotvec();Q[i,:3,:3]=Rot.from_rotvec(rel*s).as_matrix()@Q[i,:3,:3];Q[i,:3,3]=(1-s)*Q[i,:3,3]+s*t1
 return Q
def ev(Q,frames):
 vals=[]
 for i in frames:
  mm,bm=render(Q[i],i);gt=om[i][crop0[i][1]:crop0[i][1]+S,crop0[i][0]:crop0[i][0]+S];vals.append([iou(mm,gt),iou(bm,obm[i])])
 return np.array(vals)
for win in [5,9,15,21,31,41]:
 Q=make(win);ee=ev(Q,range(0,553,2));rel=np.einsum('fji,fjk->fik',Q[:-1,:3,:3],Q[1:,:3,:3]);ang=np.degrees(Rot.from_matrix(rel).magnitu …[+412 chars]
```
<details><summary>output</summary>

```
win 5 mask 0.9240379347383543 blue 0.33397865411046096 bluevis 0.6670918073288108 rotstep [0.05720185 2.54966211 4.89137028 6.12492673] racc 0.511650537471201 5.526553767693484 tacc 0.007115153804045315
win 9 mask 0.924239989741926 blue 0.3350552939448694 bluevis 0.6693776818261172 rotstep [0.04248453 2.59696885 5.0996148  6.75453044] racc 0.5333201584433036 5.625986380514977 tacc 0.003358350619129927
win 15 mask 0.9242893847523649 blue 0.33553180126629 bluevis 0.6707171676324778 rotstep [0.05672624 2.52281636 5.22412959 7.11049209] racc 0.5407669006037067 5.633269365459497 tacc 0.0019011923262748586
win 21 mask 0.9243336473853311 blue 0.3347213564552242 bluevis 0.6691035174848607 rotstep [0.05875135 2.54219252 5.26881886 7.53307688] racc 0.5381461175050724 5.865186973135471 tacc 0.0014809 …[+422 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
for n,p in [('base','/work/inputs/reconstruction.npz'),('w15','/work/scratch/cand_w15.npz')]:
 z=np.load(p);T=z['T'] if 'T' in z else z['T_world_mesh'];rel=np.einsum('fji,fjk->fik',T[:-1,:3,:3],T[1:,:3,:3]);ang=np.degrees(Rot.from_matrix(rel).magnitude());da=np.abs(np.diff(ang));inds=np.argsort(da)[-20:]
 print('\\n',n,'mean',da.mean(),'pct',np.percentile(da,[50,90,95,99,100]))
 print([(int(i),round(float(ang[i]),2),round(float(da[i]),2)) for i in inds])
PY"
```
<details><summary>output</summary>

```

 base mean 0.43656514632479826 pct [0.27799588 1.06971537 1.35825222 2.13799596 2.91977817]
[(315, 1.97, 1.54), (394, 4.99, 1.56), (384, 1.39, 1.56), (246, 2.45, 1.56), (409, 1.4, 1.6), (369, 1.15, 1.64), (238, 2.46, 1.68), (232, 0.93, 1.79), (413, 3.65, 1.79), (385, 2.95, 1.81), (273, 2.81, 1.83), (389, 4.05, 1.87), (390, 2.18, 1.88), (393, 2.95, 2.04), (341, 1.85, 2.23), (329, 4.8, 2.48), (418, 2.82, 2.5), (320, 0.32, 2.53), (414, 1.86, 2.88), (328, 1.88, 2.92)]

 w15 mean 0.5407669006037067 pct [0.01090543 1.72604311 2.41142712 4.05371665 5.63326937]
[(232, 0.75, 2.89), (210, 1.54, 2.94), (270, 1.75, 2.96), (315, 3.32, 2.97), (384, 1.17, 3.0), (238, 3.28, 3.0), (394, 5.93, 3.04), (385, 4.16, 3.19), (413, 3.75, 3.22), (273, 3.94, 3.24), (211, 4.48, 3.33), (392, 5.21, 3.33), (390, 1.17,  …[+140 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_windows.py';s=open(p).read().replace(\"for win in [5,9,15,21,31,41]:\",\"for win in [0,15]:\").replace(\"ys=savgol_filter(yaw,win,2)\",\"ys=yaw if win==0 else savgol_filter(yaw,win,2)\")
# existing make always savgol translation; win0 invalid. set tw=15
s=s.replace(\"Q=T0.copy();ys=yaw if win==0 else savgol_filter(yaw,win,2)\",\"Q=T0.copy();ys=yaw if win==0 else savgol_filter(yaw,win,2);tw=15\")
s=s.replace(\"Q[a0:a1,j,3]=savgol_filter(T0[a0:a1,j,3],win,2)+g[j]\",\"Q[a0:a1,j,3]=savgol_filter(T0[a0:a1,j,3],tw,2)+g[j]\")
s=s.replace(\"np.savez(f'/work/scratch/cand_w{win}.npz',T=Q)\",\"np.savez(f'/work/scratch/cand_w{win}.npz',T=Q)\")
open('/work/scratch/eval_norotsmooth.py','w').write(s)
PY
python /work/scratch/eval_norotsmooth.py"
```
<details><summary>output</summary>

```
win 0 mask 0.9243507893648119 blue 0.3363794718881886 bluevis 0.672044077070236 rotstep [0.05651974 2.22221968 4.26791745 4.99060623] racc 0.3249541639879699 3.521574453063928 tacc 0.0019011923262748586
win 15 mask 0.9242893847523649 blue 0.33553180126629 bluevis 0.6707171676324778 rotstep [0.05672624 2.52281636 5.22412959 7.11049209] racc 0.5407669006037067 5.633269365459497 tacc 0.0019011923262748586

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/sweep_trans.py <<'PY'
exec(open('/work/scratch/opt_static_blue.py').read().split('frames=list(range(480,553,2))')[0])
from scipy.signal import savgol_filter
a0,a1=204,480;g=np.array([-.008,.007,0.]);R0=Rot.from_matrix(T0[:a0,:3,:3]).mean().as_matrix();t0=np.median(T0[:a0,:3,3],axis=0)+g;R1=Rot.from_matrix(T0[a1:,:3,:3]).mean().as_matrix();t1=np.median(T0[a1:,:3,3],axis=0)+g
def make(tw):
 Q=T0.copy()
 for j in range(3):Q[a0:a1,j,3]=savgol_filter(T0[a0:a1,j,3],tw,2)+g[j]
 Q[:a0,:3,:3]=R0;Q[:a0,:3,3]=t0;Q[a1:,:3,:3]=R1;Q[a1:,:3,3]=t1
 for k in range(8):
  s=.5*(1-np.cos(np.pi*(k+1)/9));i=a0+k;rel=(Rot.from_matrix(R0).inv()*Rot.from_matrix(Q[i,:3,:3])).as_rotvec();Q[i,:3,:3]=Rot.from_rotvec(rel*s).as_matrix()@R0;Q[i,:3,3]=(1-s)*t0+s*Q[i,:3,3]
  i=a1-1-k;rel=(Rot.from_matrix(Q[i,:3,:3]).inv()*Rot.from_matrix(R1)).as_rotvec();Q[i,:3,:3]=Rot.from_rotvec(rel*s).as_matrix()@Q[i,:3,:3];Q[i,:3,3]=(1-s)*Q[i,:3,3]+s*t1
 return Q
def ev(Q,frames):
 vals=[]
 for i in frames:
  mm,bm=render(Q[i],i);gt=om[i][crop0[i][1]:crop0[i][1]+S,crop0[i][0]:crop0[i][0]+S];vals.append([iou(mm,gt),iou(bm,obm[i])])
 return np.array(vals)
frames=list(range(0,553,3))
for tw in [0,5,9,15,21,31,41]:
 Q=make(max(5,tw)) if tw else T0.copy()
 if tw==0:
  Q[:a0,:3,:3]=R0;Q[:a0,:3,3]=t0;Q[a1:,:3,:3]=R1;Q[a1:,:3,3]=t1
 ee=ev(Q,frames);tacc=np.linalg.norm(np.diff(Q[:,:3,3],n=2,axis=0),axis=1);print('tw',tw,'mask',ee[:,0].mean(),'blue',ee[:,1].mean(),'tacc mean/max',tacc.mean(),ta …[+49 chars]
```
<details><summary>output</summary>

```
tw 0 mask 0.9113188220177824 blue 0.32289745399529396 tacc mean/max 0.015920993483070592 0.17694326137802727
tw 5 mask 0.9242899045531152 blue 0.33601594829121323 tacc mean/max 0.007115153804045315 0.0788644729945585
tw 9 mask 0.9243650635799846 blue 0.3363131656634077 tacc mean/max 0.003358350619129927 0.06853571350479346
tw 15 mask 0.9243315450973763 blue 0.3366668330745534 tacc mean/max 0.0019011923262748586 0.0682827909641178
tw 21 mask 0.9246076283509648 blue 0.33710386354272653 tacc mean/max 0.0014809910209476077 0.06111639366328955
tw 31 mask 0.9245003827460455 blue 0.33657938404730375 tacc mean/max 0.0010546721729235479 0.05345258220639078
tw 41 mask 0.9244349961936208 blue 0.3369193733276833 tacc mean/max 0.0009638021280588595 0.07112582545855284

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_eval.py <<'PY'
exec(open('/work/scratch/opt_static_blue.py').read().split('frames=list(range(480,553,2))')[0])
from scipy.signal import savgol_filter
a0,a1=204,480;g=np.array([-.008,.007,0.]);R0=Rot.from_matrix(T0[:a0,:3,:3]).mean().as_matrix();t0=np.median(T0[:a0,:3,3],axis=0)+g;R1=Rot.from_matrix(T0[a1:,:3,:3]).mean().as_matrix();t1=np.median(T0[a1:,:3,3],axis=0)+g
Q=T0.copy()
for j in range(3):Q[a0:a1,j,3]=savgol_filter(T0[a0:a1,j,3],21,2)+g[j]
Q[:a0,:3,:3]=R0;Q[:a0,:3,3]=t0;Q[a1:,:3,:3]=R1;Q[a1:,:3,3]=t1
for k in range(8):
 s=.5*(1-np.cos(np.pi*(k+1)/9));i=a0+k;rel=(Rot.from_matrix(R0).inv()*Rot.from_matrix(Q[i,:3,:3])).as_rotvec();Q[i,:3,:3]=Rot.from_rotvec(rel*s).as_matrix()@R0;Q[i,:3,3]=(1-s)*t0+s*Q[i,:3,3]
 i=a1-1-k;rel=(Rot.from_matrix(Q[i,:3,:3]).inv()*Rot.from_matrix(R1)).as_rotvec();Q[i,:3,:3]=Rot.from_rotvec(rel*s).as_matrix()@Q[i,:3,:3];Q[i,:3,3]=(1-s)*Q[i,:3,3]+s*t1
np.save('/work/scratch/final_T.npy',Q)
def metrics(T):
 vals=[]
 for i in range(553):
  mm,bm=render(T[i],i);gt=om[i][crop0[i][1]:crop0[i][1]+S,crop0[i][0]:crop0[i][0]+S];obsb=obm[i]
  inter=(mm&gt).sum();union=(mm|gt).sum();mi=inter/max(1,union);bi=iou(bm,obsb) if obsb.sum() else np.nan
  yy,xx=np.where(mm);gy,gx=np.where(gt);vals.append([mi,bi,xx.mean()-gx.mean(),yy.mean()-gy.mean(),mm.sum(),gt.sum(),bm.sum(),obsb.sum()])
 return np.array(vals)
B=metrics(T0);R=metrics(Q);np.savez('/work/scratch/final_metrics.npz',base=B,result=R)
phases=[('rest_precontac …[+788 chars]
```
<details><summary>output</summary>

```
phase baseline vs result
/work/scratch/final_eval.py:22: RuntimeWarning: Mean of empty slice
  print(n,a,b,'mask',B[a:b,0].mean(),R[a:b,0].mean(),'blue',np.nanmean(B[a:b,1]),np.nanmean(R[a:b,1]),'centx',B[a:b,2].mean(),R[a:b,2].mean(),'centy',B[a:b,3].mean(),R[a:b,3].mean())
rest_precontact 0 140 mask 0.8850607525799731 0.9260787513783006 blue nan nan centx 2.25663858331991 0.16508368562443373 centy -0.9580733543648546 1.8584266310890736
rest_contact 140 204 mask 0.9025198774661398 0.9290865457756476 blue nan nan centx 1.9714099370129836 0.03767873062164662 centy -0.38319460477499234 1.2391863179131333
rotate 204 420 mask 0.8899253679837609 0.9111443759568759 blue 0.5753693682070946 0.6064306461521204 centx 2.036772760453571 -0.07660207830034357 centy -0.7034845615715414 1.1722846527471051 …[+994 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];yaw=np.unwrap(np.arctan2(T[:,2,0],T[:,0,0]));yaw=np.degrees(yaw-yaw[0]);
for i in range(185,221): print(i,round(yaw[i],3),np.round(T[i,:3,3],4))
PY"
```
<details><summary>output</summary>

```
185 -0.869 [0.1564 0.2988 3.1426]
186 -0.768 [0.1583 0.3012 3.1577]
187 -1.082 [0.1575 0.3005 3.1539]
188 -0.505 [0.1577 0.3023 3.1541]
189 -0.299 [0.1563 0.3008 3.1508]
190 -1.258 [0.1597 0.3022 3.1675]
191 -1.296 [0.1597 0.3003 3.151 ]
192 -1.107 [0.1556 0.299  3.1489]
193 -0.861 [0.157  0.2997 3.1505]
194 -0.782 [0.1577 0.3009 3.1655]
195 -0.958 [0.1559 0.2998 3.149 ]
196 -0.932 [0.1561 0.2999 3.1467]
197 -1.14 [0.1567 0.2993 3.1461]
198 -1.074 [0.1576 0.3021 3.1648]
199 -0.702 [0.1579 0.3012 3.1455]
200 -1.017 [0.1577 0.3007 3.1442]
201 -1.072 [0.1583 0.3009 3.1452]
202 -1.309 [0.1578 0.3027 3.155 ]
203 -1.379 [0.158  0.3018 3.152 ]
204 -1.055 [0.1575 0.3014 3.1452]
205 -1.415 [0.1574 0.3016 3.1453]
206 -1.164 [0.1573 0.3023 3.1481]
207 -1.545 [0.1576 0.302  3.1514]
208 -1.62 [0.157  0 …[+424 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];
for i in range(474,501): print(i,np.round(T[i,:3,3],4))
PY"
```
<details><summary>output</summary>

```
474 [0.1641 0.2878 3.1584]
475 [0.1627 0.2857 3.148 ]
476 [0.1631 0.2854 3.1453]
477 [0.1644 0.2861 3.1463]
478 [0.165  0.2869 3.1565]
479 [0.1615 0.2841 3.1081]
480 [0.1593 0.2823 3.1051]
481 [0.1582 0.2772 3.0701]
482 [0.1613 0.2794 3.1014]
483 [0.156  0.2735 3.0369]
484 [0.1588 0.2773 3.0728]
485 [0.1601 0.2799 3.0972]
486 [0.1616 0.2792 3.0995]
487 [0.1618 0.2791 3.1036]
488 [0.1606 0.2791 3.1295]
489 [0.1597 0.2786 3.0822]
490 [0.159  0.28   3.1208]
491 [0.1565 0.2779 3.0498]
492 [0.1607 0.2822 3.1277]
493 [0.1597 0.2807 3.1061]
494 [0.16   0.2827 3.1391]
495 [0.1573 0.2832 3.1078]
496 [0.1589 0.2842 3.1322]
497 [0.159  0.2828 3.1279]
498 [0.1582 0.2825 3.1226]
499 [0.1576 0.2838 3.108 ]
500 [0.1596 0.284  3.103 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
for a,b in [(0,120),(0,140),(0,204),(480,553),(490,553),(460,553)]:
 rm=R.from_matrix(T[a:b,:3,:3]).mean();tm=np.median(T[a:b,:3,3],axis=0);print(a,b,'rotvec',rm.as_rotvec(),'t',tm)
PY"
```
<details><summary>output</summary>

```
0 120 rotvec [-1.05784837  1.32191238  1.35529806] t [0.15615627 0.28976476 3.1187487 ]
0 140 rotvec [-1.05651767  1.32080931  1.35641927] t [0.15730168 0.29225123 3.131585  ]
0 204 rotvec [-1.05375516  1.32082075  1.35489487] t [0.15748668 0.29786712 3.1413388 ]
480 553 rotvec [-1.38559184  0.65011454  0.67559794] t [0.16108833 0.28144488 3.0952344 ]
490 553 rotvec [-1.38229962  0.64948993  0.67688251] t [0.1612618 0.2817211 3.0949502]
460 553 rotvec [-1.39018305  0.6491771   0.67455994] t [0.16149785 0.28216332 3.0993392 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
for a,b in [(500,553),(510,553),(520,553),(480,553)]:
 print(a,b,'tmean',T[a:b,:3,3].mean(0),'median',np.median(T[a:b,:3,3],axis=0),'rot',R.from_matrix(T[a:b,:3,:3]).mean().as_rotvec())
PY"
```
<details><summary>output</summary>

```
500 553 tmean [0.16177097 0.2807904  3.0931299 ] median [0.1615764  0.28165802 3.0930665 ] rot [-1.37929054  0.64838447  0.6785461 ]
510 553 tmean [0.16199559 0.2804148  3.0919952 ] median [0.1619503  0.28165802 3.09211   ] rot [-1.37515716  0.64546112  0.67831652]
520 553 tmean [0.16176555 0.27974853 3.0890007 ] median [0.16164663 0.2800775  3.0890186 ] rot [-1.37362831  0.64317266  0.67794086]
480 553 tmean [0.16106886 0.28065097 3.0955663 ] median [0.16108833 0.28144488 3.0952344 ] rot [-1.38559184  0.65011454  0.67559794]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];Tr=np.load('/work/scratch/final_T.npy')
for n,T in [('base',Tb),('res',Tr)]:
 rel=np.einsum('fji,fjk->fik',T[:-1,:3,:3],T[1:,:3,:3]);ang=np.degrees(R.from_matrix(rel).magnitude());steps=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);tacc=np.linalg.norm(np.diff(T[:,:3,3],n=2,axis=0),axis=1)
 print(n,'rot p',np.percentile(ang,[50,90,99,100]),'trans p',np.percentile(steps,[50,90,99,100]),'tacc',tacc.mean(),tacc.max())
 for a,b in [(0,204),(480,553)]:
  yaw=np.unwrap(np.arctan2(T[a:b,2,0],T[a:b,0,0]));print(' static',a,b,'yaw std/range',np.degrees(yaw.std()),np.degrees(np.ptp(yaw)),'tstd',T[a:b,:3,3].std(0),'trange',np.ptp(T[a:b,:3,3],axis=0))
PY"
```
<details><summary>output</summary>

```
base rot p [0.70995281 2.21627586 4.07153155 4.99060585] trans p [0.0075075  0.04277331 0.08549317 0.09327658] tacc 0.026934579 0.17694327
 static 0 204 yaw std/range 0.4287913 2.3574603 tstd [0.00278064 0.00603233 0.0321503 ] trange [0.01423845 0.02414832 0.17499876]
 static 480 553 yaw std/range 0.38662666 1.6356117 tstd [0.00174854 0.00290996 0.0168842 ] trange [0.00790279 0.01233599 0.10218263]
res rot p [0.05651974 2.22221968 4.26791745 4.99060623] trans p [5.73456366e-05 4.35735324e-03 1.01519724e-02 5.62003806e-02] tacc 0.0014809910209476077 0.06111639366328955
 static 0 204 yaw std/range 1.2722218725854067e-14 0.0 tstd [6.38378239e-16 7.21644966e-16 0.00000000e+00] trange [0. 0. 0.]
 static 480 553 yaw std/range 0.0 0.0 tstd [2.49800181e-16 1.66533454e-16 0.00000000e+00] trange [0. …[+8 chars]
```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent world-space rice-cooker trajectory.

The released baseline is already a strong image-based estimate.  This solver
uses its dynamic rotation, but replaces estimator jitter in the two physical
rest phases, removes depth/tracker jitter from the manipulation trajectory, and
applies one small world-space correction supported by pooled silhouette
evidence.  It is deterministic and uses only the released inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# Boundaries verified from the baseline rotation-speed, the right-hand contact
# signal, and the object'"'s screen-space motion:
#   * frames 0..203: object stationary (the hand arrives around frame 140 but
#     the object remains fixed until the clockwise turn begins around 209);
#   * frames 204..479: turn plus the final placement translation;
#   * frames 480..552: object stationary on the table.
STATIC_PHASES = ((0, 204), (480, 553))
MOVING_START = 204
MOVING_STOP = 480

# Pooled-fit correction in world metres.  The baseline projection is offset by
# approximately +2.1 px in x and -0.9 px in y.  Optimising a constant silhouette
# displacement over the two static clips gives the same correction in both:
# about -8 mm along world x and +7 mm along world y.  Depth is deliberat …[+4633 chars]
```
<details><summary>output</summary>

```
/usr/bin/bash: line 140: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cd /work && TIMEFORMAT='elapsed=%3R sec'; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
wrote /work/outputs/object_poses_world.npz with T_world_mesh shape (553, 4, 4) and dtype float64
elapsed=0.883 sec

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
O=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];Q=np.load('/work/scratch/final_T.npy')
print(O.shape,O.dtype,'finite',np.isfinite(O).all(),'allclose scratch',np.allclose(O,Q,atol=1e-10),'maxdiff',np.abs(O-Q).max())
print('det err',np.abs(np.linalg.det(O[:,:3,:3])-1).max(),'orth err',np.abs(np.einsum('fji,fjk->fik',O[:,:3,:3],O[:,:3,:3])-np.eye(3)).max())
print('bottom row unique',np.unique(O[:,3,:],axis=0))
print('first,last',O[0],O[-1])
PY"
```
<details><summary>output</summary>

```
(553, 4, 4) float64 finite True allclose scratch False maxdiff 9.274489912836081e-06
det err 1.1102230246251565e-15 orth err 8.881784197001252e-16
bottom row unique [[0. 0. 0. 1.]]
first,last [[-0.19110484 -0.98110905  0.03006604  0.14948668]
 [ 0.05504026  0.01987145  0.99828638  0.30486712]
 [-0.98002526  0.1924322   0.05020297  3.14133894]
 [ 0.          0.          0.          1.        ]] [[ 0.65387866 -0.75638551  0.01799045  0.15308833]
 [ 0.04704554  0.06437888  0.99681597  0.28844488]
 [-0.75513536 -0.65095032  0.07768054  3.09523439]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/final_metrics.npz');B=z['base'];R=z['result']
for n,x in [('base',B),('res',R)]: print(n,'cent',x[:,2].mean(),x[:,3].mean(),'abs',np.abs(x[:,2]).mean(),np.abs(x[:,3]).mean())
PY"
```
<details><summary>output</summary>

```
base cent 2.0789654564966487 -0.9376806838308639 abs 2.0789654564966487 0.9848983900778485
res cent -0.01333722508603844 1.1721363963046612 abs 0.25392181293699556 1.1736038985459996

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'git -C /code/reconstruction rev-parse HEAD 2>/dev/null || true; git -C /code/reconstruction log -1 --oneline 2>/dev/null || true; python --version'
```
<details><summary>output</summary>

```
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80
7c0d3b9 Add challenge FAQ links to contact and submission sections (#191)
Python 3.11.10

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy,scipy,h5py,trimesh,cv2,torch,nvdiffrast,open3d,pytorch3d,PIL
for n,m in [('numpy',numpy),('scipy',scipy),('h5py',h5py),('trimesh',trimesh),('opencv',cv2),('torch',torch),('nvdiffrast',nvdiffrast),('open3d',open3d),('pytorch3d',pytorch3d),('Pillow',PIL)]: print(n,getattr(m,'__version__','unknown'))
PY"
```
<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
h5py 3.16.0
trimesh 5.1.1
opencv 4.11.0
torch 2.5.1+cu124
nvdiffrast 0.4.0
open3d 0.20.0
pytorch3d 0.7.9
Pillow 10.2.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Rice-cooker trajectory recovery

## Result

"'`object_poses_world.npz` contains `T_world_mesh` as a `(553, 4, 4)` float64 array.
It maps the released `object_mesh.glb` coordinates to the released world frame
for every frame of `video.mp4`.

The released baseline is strong in this episode: its dynamic rotation follows
the real clockwise turn, and it does **not** contain a 180-degree segment flip.
The improvement therefore comes from enforcing the physical trajectory phases,
pooling evidence inside each rest phase, removing per-frame depth jitter, and
correcting a small systematic image-plane offset.

## Phases found

The phase boundaries were selected from the baseline unwrapped rotation speed,
the right-hand contact probability, the object screen-space motion, and visual
checks of the blue side handle. They are encoded as explicit constants in
`solve.py` with the same reasons.

| Frames | Phase | Evidence |
|---|---|---|
| 0-139 | Rest before contact | Constant image silhouette; no right-hand contact; baseline yaw varies by only 2.36 deg and its apparent depth wanders by 17.5 cm. |
| 140-203 | Held, still at rest | Right-hand contact becomes stable, but the object does not turn until about frame 209. Kept in the same constant-pose rest phase. |
| 204-419 | Clockwise rotation | Baseline yaw changes continuously by about 308 deg. The blue handle enters at the right image edge around frame 280, sweeps across the front, and …[+7228 chars]
```
<details><summary>output</summary>

```
total 100K
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 22:14 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 21:52 ..
-rw-rw-r-- 1 hanzhou hanzhou 6.6K Oct 20 22:14 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.8K Oct 20 22:14 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou  70K Oct 20 22:14 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou 5.8K Oct 20 22:14 solve.py
  120 /work/outputs/REPORT.md
   46 /work/outputs/TOOLS.md
  136 /work/outputs/solve.py
  302 total

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cd /tmp && python /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/repro_check.npz && python - <<'PY'
import numpy as np, hashlib
p='/work/outputs/object_poses_world.npz';q='/work/scratch/repro_check.npz'
a=np.load(p)['T_world_mesh'];b=np.load(q)['T_world_mesh']
print('same array:',np.array_equal(a,b),'shape:',a.shape,'dtype:',a.dtype,'finite:',np.isfinite(a).all())
print('sha256:',hashlib.sha256(open(p,'rb').read()).hexdigest())
print('translation bounds:',a[:,:3,3].min(0),a[:,:3,3].max(0))
PY"
```
<details><summary>output</summary>

```
wrote /work/scratch/repro_check.npz with T_world_mesh shape (553, 4, 4) and dtype float64
same array: True shape: (553, 4, 4) dtype: float64 finite: True
sha256: ce11fbbc56725a16614f3852d48adf22fbe6fa26ab42e9133db4c00398c5bfb3
translation bounds: [0.14216211 0.28844488 3.09523439] [0.16015835 0.30715529 3.20030358]

```
</details>

**Agent:** Completed the task. Required outputs are under `/work/outputs`:

- `object_poses_world.npz` — `(553, 4, 4)` float64 `T_world_mesh`
- `solve.py` — deterministic offline solver, verified to reproduce the output bit-for-bit in 0.9 s
- `REPORT.md` — phases, method, validation table, confidence, and run time
- `TOOLS.md` — tools and versions used

Validation mask IoU improved from `0.8882` to `0.9244` overall, with improvements in every phase. Static-phase depth jitter was removed, translation acceleration dropped by about 18×, and the blue-handle track independently confirmed the baseline orientation is not 180° flipped.
