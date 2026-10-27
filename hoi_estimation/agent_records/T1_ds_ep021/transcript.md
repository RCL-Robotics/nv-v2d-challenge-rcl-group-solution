# T1_ds_ep021

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T18:04:45Z

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
/usr/bin/bash -c "sed -n '1,80p' /work/AGENT_TASK.md && printf '\\n---80-160---\\n' && sed -n '80,180p' /work/AGENT_TASK.md"
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
and translation both matter; a  …[+9921 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "wc -l /work/AGENT_TASK.md && sed -n '45,92p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
120 /work/AGENT_TASK.md
   smoothness / constant-velocity terms on SE(3), and rigid hand-object attachment while carried.
5. **Anchors and propagation**: find frames where the pose is unambiguous (object large, little occluded, distinctive
   view, high evidence margin between hypotheses) and propagate forward and backward from them through the ambiguous
   segments, rather than trusting every frame equally.

## Validate without ground truth
You have no ground truth and must not try to get it. Build your own checks and make decisions with them:
measures of how well a trajectory explains the video (held-out frames, mask IoU, photometric / flow residuals,
temporal smoothness, physical plausibility such as no floating or penetrating the floor/human), with a table baseline
vs. your result per  …[+3278 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%f %s bytes\\n' | sort && echo '--- task' && cat /work/inputs/task.txt && echo '--- episode' && cat /work/inputs/episode.json && echo '--- camera' && cat /work/inputs/camera.json && echo '--- plane' && cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
camera.json 820 bytes
episode.json 69 bytes
frame0_reference_human.npz 191486 bytes
ground_plane.json 399 bytes
masks.h5 7107337 bytes
object_mesh.glb 14269716 bytes
reconstruction.npz 34099235 bytes
task.txt 80 bytes
video.mp4 20158142 bytes
--- task
Walk over to the tall bar stool and pull the tall bar stool away from the desk.
--- episode
{
 "episode_index": 21,
 "object": "tall_bar_stool",
 "frames": 678
}--- camera
{
 "K": [
  [
   819.9996337890625,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.9996337890625,
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
   804.0242919921875,
   0.0,
   768.0
  ],
  [
   0.0,
   804.0242919921875,
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
   0.054485605610393305,
   0.07374494367760444,
   - …[+853 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
p='/work/inputs/reconstruction.npz'
z=np.load(p)
for k in z.files:
 a=z[k]
 print(k,a.shape,a.dtype)
 if k=='T_world_mesh':
  for i in [0,1,len(a)-1]:
   print(i, a[i])
PY"
```
<details><summary>output</summary>

```
T_world_mesh (678, 4, 4) float32
0 [[-0.66395795  0.747123   -0.03109885 -0.06167061]
 [-0.07739515 -0.0272956   0.99662685  0.66982305]
 [ 0.74375385  0.66412514  0.07594682  3.360184  ]
 [ 0.          0.          0.          1.        ]]
1 [[-0.6663891   0.7450008  -0.02999119 -0.05439661]
 [-0.07849816 -0.03010105  0.9964597   0.66801953]
 [ 0.74146056  0.66638404  0.07854019  3.3581145 ]
 [ 0.          0.          0.          1.        ]]
677 [[-0.14721839 -0.9889977   0.01441855  0.12755965]
 [ 0.03464025  0.00941331  0.99934787  0.726175  ]
 [-0.9884965   0.14762041  0.03287303  2.2718482 ]
 [ 0.          0.          0.          1.        ]]
T_cam_mesh_foundationpose (678, 4, 4) float32
T_cam_mesh_coconet (678, 4, 4) float32
human_vertices_world (678, 4610, 3) float32
human_keypoints …[+69 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -100"
```
<details><summary>output</summary>

```
/pipeline/.stages/03_human_initialization.json 2295
/pipeline/.stages/01_depth.json 1911
/pipeline/.stages/02_export.json 2160
/pipeline/.stages/04_depth_alignment.json 2057
/pipeline/.stages/05_object_pose.json 1898
/pipeline/.stages/06_coconet.json 3080
/pipeline/.stages/07_refinement.json 2299
/pipeline/.stages/08_render.json 2820
/pipeline/depth/moge2/intrinsics.pkl 72275
/pipeline/depth/moge2/raw.npy 2399404160
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/depth_report.json 1875
/pipeline/depth/moge2/aligned.h5 750948936
/pipeline/mhr_init/episode_000021.stats.pkl 6607
/pipeline/mhr_init/episode_000021.sam3d.h5 694013
/pipeline/mhr_init/episode_000021.pkl 3026295
/pipeline/foundationpose/episode_000021.pkl 194945
/pipeline/inference/.materialized_inputs …[+480 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, cv2, os
p='/work/inputs/masks.h5'
with h5py.File(p,'r') as f:
 for k in f.keys():
  print(k,f[k].shape,f[k].dtype)
v=cv2.VideoCapture('/work/inputs/video.mp4')
print('video',v.get(cv2.CAP_PROP_FRAME_COUNT),v.get(cv2.CAP_PROP_FPS),v.get(cv2.CAP_PROP_FRAME_WIDTH),v.get(cv2.CAP_PROP_FRAME_HEIGHT))
PY"
```
<details><summary>output</summary>

```
object (678, 1152, 1536) bool
person (678, 1152, 1536) bool
video 678.0 30.0 1536.0 1152.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh'].astype(float)
p=T[:,:3,3]
r=R.from_matrix(T[:,:3,:3]).as_rotvec()
# speed and jumps
dp=np.linalg.norm(np.diff(p,axis=0),axis=1)
dr=np.linalg.norm(np.diff(r,axis=0),axis=1)
for name,a,idx in [('dp',dp,np.argsort(dp)[-20:]),('dr',dr,np.argsort(dr)[-20:])]:
 print(name,'median',np.median(a),'p90',np.percentile(a,90),'max',a.max())
 print(sorted([(int(i),round(float(a[i]),4)) for i in idx]))
print('pos range',p.min(0),p.max(0))
# report every 15 poses pos, rotvec
for i in range(0,len(T),15):
 print(i, np.round(p[i],3), np.round(r[i],2))
PY"
```
<details><summary>output</summary>

```
dp median 0.008335034808867698 p90 0.019407529447576746 max 0.0543764458428677
[(238, 0.0275), (409, 0.0258), (423, 0.0267), (425, 0.0256), (431, 0.0255), (442, 0.0472), (444, 0.0278), (448, 0.0351), (452, 0.0475), (453, 0.0317), (458, 0.029), (460, 0.0295), (473, 0.0416), (529, 0.0544), (532, 0.0345), (533, 0.0382), (537, 0.0345), (544, 0.0434), (555, 0.0257), (559, 0.0261)]
dr median 0.008496637303963852 p90 0.04525480782082036 max 6.2668379936824286
[(174, 0.0974), (426, 0.0741), (432, 0.0785), (452, 0.0774), (530, 0.1303), (533, 0.1355), (534, 6.199), (535, 6.2668), (536, 6.2447), (539, 0.0848), (540, 0.0971), (541, 0.0851), (542, 0.0792), (543, 0.0829), (544, 0.0914), (546, 0.0862), (547, 0.0834), (548, 0.0885), (549, 0.0932), (553, 0.0769)]
pos range [-0.08331043  0.63769674  2.26321 …[+2122 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
p='/work/inputs/object_mesh.glb'
s=trimesh.load(p, force='scene', process=False)
g=s.to_geometry()
print('scene',s, 'geometry', type(g), len(g) if hasattr(g,'__len__') else '')
if isinstance(g, trimesh.Scene):
 print(g.geometry)
 g=list(g.geometry.values())
else: g=[g] if not isinstance(g,(list,tuple,np.ndarray)) else g
for i,x in enumerate(g):
 print(i,type(x),x.vertices.shape,x.faces.shape,x.bounds,x.extents,x.is_watertight)
 print('visual',type(x.visual), getattr(x.visual,'material',None))
 print('uv',getattr(x.visual,'uv',None))
PY"
```
<details><summary>output</summary>

```
scene <trimesh.Scene(len(geometry)=1)> geometry <class 'trimesh.base.Trimesh'> 
0 <class 'trimesh.base.Trimesh'> (51247, 3) (95759, 3) [[-0.20217043 -0.23005383 -0.35287058]
 [ 0.26516873  0.22619514  0.53656369]] [0.46733916 0.45624897 0.88943428] False
visual <class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x75004eb074d0>
uv [[0.82218701 0.83954698]
 [0.82177699 0.83927602]
 [0.822779   0.83936501]
 ...
 [0.040383   0.01924098]
 [0.92341    0.04879498]
 [0.98682302 0.86608398]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline/export/episode_000021 -maxdepth 2 -type f -printf '%p %s\\n'; echo '--- report'; cat /pipeline/export/episode_000021/wild_export.json; echo '--- stages'; for f in /pipeline/.stages/*.json; do echo ==== "'$f; cat $f; done'
```
<details><summary>output</summary>

```
/pipeline/export/episode_000021/images/front_stereo_camera_left.h5 564143864
/pipeline/export/episode_000021/images/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000021/human_masks/front_stereo_camera_left.h5 16939776
/pipeline/export/episode_000021/human_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000021/object_masks/front_stereo_camera_left.h5 17254653
/pipeline/export/episode_000021/object_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000021/edex 563
/pipeline/export/episode_000021/object_mesh/output_aligned.glb 14269996
/pipeline/export/episode_000021/wild_export.json 2885
--- report
{
  "aligned_object_extents_m": [
    0.4562727212905884,
    0.4669312834739685,
    0.8900071382522 …[+21651 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 o=f['object'][:]; p=f['person'][:]
a=o.sum((1,2)); b=p.sum((1,2));
for i in range(0,678,15): print(i,a[i],b[i],round(a[i]/(1536*1152)*100,3))
PY"
```
<details><summary>output</summary>

```
0 15708 25452 0.888
15 15629 29165 0.883
30 15740 28565 0.89
45 15796 37090 0.893
60 15764 45662 0.891
75 15794 46549 0.893
90 15498 40656 0.876
105 15894 47764 0.898
120 15600 47776 0.882
135 16445 44056 0.929
150 16704 41000 0.944
165 15934 47364 0.9
180 18656 45511 1.054
195 23334 44120 1.319
210 24684 44494 1.395
225 25593 47611 1.446
240 28934 44768 1.635
255 28590 52809 1.616
270 23863 62638 1.349
285 24558 60746 1.388
300 23895 71408 1.35
315 24174 74828 1.366
330 22673 73577 1.281
345 19848 77046 1.122
360 20924 76060 1.182
375 23931 77630 1.352
390 20772 82399 1.174
405 22077 82858 1.248
420 16056 77874 0.907
435 8603 82809 0.486
450 6469 80473 0.366
465 10752 62731 0.608
480 10199 53006 0.576
495 10939 49920 0.618
510 11600 50838 0.656
525 6768 56972 0.382
540 7755 64119 0.438
55 …[+196 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/frames /work/scratch/vis && ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -vf \"select='not(mod(n,15))',scale=384:-1,tile=5x10\" -vsync vfr /work/scratch/vis/contact15.jpg && ls -lh /work/scratch/vis/contact15.jpg"
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 275K Oct 20 21:59 /work/scratch/vis/contact15.jpg

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print('material',m.visual.material)
print('image',m.visual.material.image, m.visual.material.image.size if m.visual.material.image else None)
print('metallic',m.visual.material.metallicFactor,'rough',m.visual.material.roughnessFactor)
print('uv range',m.visual.uv.min(0),m.visual.uv.max(0))
print('volume area',m.volume,m.area)
# principal axes
import numpy as np
print('principal',m.principal_inertia_components,m.principal_inertia_vectors)
PY"
```
<details><summary>output</summary>

```
material <trimesh.visual.material.PBRMaterial object at 0x72ac0ef15a90>
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['pyrender','nvdiffrast.torch','open3d','cv2','trimesh','torch','kornia','imageio']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__',''))
 except Exception as e: print(m,'NO',repr(e))
PY"
```
<details><summary>output</summary>

```
pyrender OK 0.1.45
nvdiffrast.torch OK 0.4.0
open3d OK 0.20.0
cv2 OK 4.11.0
trimesh OK 5.1.1
torch OK 2.5.1+cu124
kornia OK 0.8.3
imageio OK 2.38.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, numpy as np, cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as f:
 for ix in [0,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,650,677]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,ix); ok,im=cap.read()
  o=f['object'][ix]; p=f['person'][ix]
  ys,xs=np.where(o); yp,xp=np.where(p)
  print(ix,'obj bbox', (xs.min(),ys.min(),xs.max(),ys.max()) if len(xs) else None,'cent', (xs.mean(),ys.mean()) if len(xs) else None,'person bbox',(xp.min(),yp.min(),xp.max(),yp.max()))
  # crop union bbox with margin, overlay object red and person blue boundaries
  if len(xs):
   x0=max(0,min(xs.min(),xp.min())-80); x1=min(1535,max(xs.max(),xp.max())+80); y0=max(0,min(ys.min(),yp.min())-80); y1=min(1151,max(ys.max(),yp.max())+80)
  else: x0,y0,x1,y1=0,0,1535,1151
  crop=im[y0:y1,x0:x1].copy(); oc=o[y0:y1,x0:x1]; pc=p[y0:y1,x0:x1]
  crop[oc]=0.55*crop[oc]+0.45*np.array([255,0,0],dtype=np.uint8)
  cont,_=cv2.findContours(pc.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(crop,cont,-1,(255,0,0),2)
  cv2.imwrite(f'/work/scratch/vis/frame_{ix:03d}.jpg',crop)
PY
ls /work/scratch/vis/frame_*.jpg | head"
```
<details><summary>output</summary>

```
0 obj bbox (759, 648, 906, 905) cent (824.5516297428062, 741.816653934301) person bbox (682, 448, 805, 814)
90 obj bbox (759, 647, 906, 905) cent (825.0251000129049, 742.7688734030197) person bbox (626, 427, 765, 941)
120 obj bbox (757, 649, 906, 906) cent (824.3383333333334, 743.2982692307693) person bbox (638, 423, 823, 937)
150 obj bbox (723, 661, 884, 925) cent (793.651400862069, 762.7651460727969) person bbox (601, 429, 770, 931)
180 obj bbox (644, 656, 809, 961) cent (720.0497427101201, 780.299903516295) person bbox (542, 429, 727, 926)
210 obj bbox (633, 659, 819, 998) cent (714.7152406417113, 788.7192513368984) person bbox (561, 430, 722, 926)
240 obj bbox (728, 674, 928, 1050) cent (818.0864035390889, 823.4367526093869) person bbox (636, 430, 808, 954)
270 obj bbox (841, 706, 1027 …[+1854 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_test.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, cv2, os
import nvdiffrast.torch as dr
np.set_printoptions(suppress=True)
im=torch.cuda.is_available(); print('cuda',im)
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32); Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(np.float32)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda')
F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
glctx=dr.RasterizeCudaContext()
Kt=torch.tensor(K,device='cuda')
Twc_i=np.linalg.inv(Twc)

def render(Twm):
    M=torch.tensor(Twc_i@Twm,dtype=torch.float32,device='cuda')
    v=V@M[:3,:3].T+M[:3,3]
    zz=v[:,2].clamp_min(1e-5)
    u=Kt[0,0]*v[:,0]/zz+Kt[0,2]; vv=Kt[1,1]*v[:,1]/zz+Kt[1,2]
    pos=torch.stack([2*u/W-1,1-2*vv/H,zz],-1)[None]
    rast,tri=dr.rasterize(glctx,pos,F,resolution=[H,W])
    mask=(tri[0,:,:,0]>=0).cpu().numpy()
    return mask
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [0,1,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,525,530,534,535,540,550,570,600,650,677]:
  mask=render(T[i]); gt=f['object'][i]
  inter=(mask&gt).sum(); union=(mask|gt).sum(); iou=inter/union if union else 0
  print(i,'iou',round(iou,4),'mask',mask.sum(),'gt',gt.sum(),'prec',round(inter …[+301 chars]
```
<details><summary>output</summary>

```
cuda True
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 26, in <module>
    mask=render(T[i]); gt=f['object'][i]
         ^^^^^^^^^^^^
  File "/work/scratch/render_test.py", line 21, in render
    rast,tri=dr.rasterize(glctx,pos,F,resolution=[H,W])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+395 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/torch.stack(\\[2\\*u\\/W-1,1-2\\*vv\\/H,zz\\],-1)/torch.stack([2*u\\/W-1,1-2*vv\\/H,zz,torch.ones_like(zz)],-1)/\" /work/scratch/render_test.py && python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
cuda True
0 iou 0.0089 mask 1769472 gt 15708 prec 0.009 rec 1.0
1 iou 0.0087 mask 1769472 gt 15355 prec 0.009 rec 1.0
90 iou 0.0088 mask 1769472 gt 15498 prec 0.009 rec 1.0
120 iou 0.0088 mask 1769472 gt 15600 prec 0.009 rec 1.0
150 iou 0.0094 mask 1769472 gt 16704 prec 0.009 rec 1.0
180 iou 0.0105 mask 1769472 gt 18656 prec 0.011 rec 1.0
210 iou 0.0139 mask 1769472 gt 24684 prec 0.014 rec 1.0
240 iou 0.0164 mask 1769472 gt 28934 prec 0.016 rec 1.0
270 iou 0.0135 mask 1769472 gt 23863 prec 0.013 rec 1.0
300 iou 0.0135 mask 1769472 gt 23895 prec 0.014 rec 1.0
330 iou 0.0128 mask 1769472 gt 22673 prec 0.013 rec 1.0
360 iou 0.0118 mask 1769472 gt 20924 prec 0.012 rec 1.0
390 iou 0.0117 mask 1769472 gt 20772 prec 0.012 rec 1.0
420 iou 0.0091 mask 1769472 gt 16056 prec 0.009 rec 1.0
450 iou 0.0 …[+707 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# quick modify print
exec(open('/work/scratch/render_test.py').read().replace(\"mask=(tri[0,:,:,0]>=0).cpu().numpy()\",\"print('tri',tri.dtype,tri.shape,tri.min().item(),tri.max().item(),(tri[0,:,:,0]>=0).sum().item()); mask=(tri[0,:,:,0]>=0).cpu().numpy()\").replace(\"for i in [0,1,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,525,530,534,535,540,550,570,600,650,677]:\",\"for i in [0]:\") )
PY"
```
<details><summary>output</summary>

```
cuda True
tri torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 0.0 1769472
0 iou 0.0089 mask 1769472 gt 15708 prec 0.009 rec 1.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.getsource(dr.rasterize))
PY"
```
<details><summary>output</summary>

```
def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    '''Rasterize triangles.

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
        …[+1402 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/rast,tri=dr.rasterize/rast,tri=dr.rasterize/; s/mask=(tri\\[0,:,:,0\\]>=0)/mask=(rast[0,:,:,3]>=0)/' /work/scratch/render_test.py && python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
cuda True
0 iou 0.0089 mask 1769472 gt 15708 prec 0.009 rec 1.0
1 iou 0.0087 mask 1769472 gt 15355 prec 0.009 rec 1.0
90 iou 0.0088 mask 1769472 gt 15498 prec 0.009 rec 1.0
120 iou 0.0088 mask 1769472 gt 15600 prec 0.009 rec 1.0
150 iou 0.0094 mask 1769472 gt 16704 prec 0.009 rec 1.0
180 iou 0.0105 mask 1769472 gt 18656 prec 0.011 rec 1.0
210 iou 0.0139 mask 1769472 gt 24684 prec 0.014 rec 1.0
240 iou 0.0164 mask 1769472 gt 28934 prec 0.016 rec 1.0
270 iou 0.0135 mask 1769472 gt 23863 prec 0.013 rec 1.0
300 iou 0.0135 mask 1769472 gt 23895 prec 0.014 rec 1.0
330 iou 0.0128 mask 1769472 gt 22673 prec 0.013 rec 1.0
360 iou 0.0118 mask 1769472 gt 20924 prec 0.012 rec 1.0
390 iou 0.0117 mask 1769472 gt 20772 prec 0.012 rec 1.0
420 iou 0.0091 mask 1769472 gt 16056 prec 0.009 rec 1.0
450 iou 0.0 …[+707 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'mask' /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
22:    mask=(rast[0,:,:,3]>=0).cpu().numpy()
23:    return mask
24:with h5py.File('/work/inputs/masks.h5','r') as f:
26:  mask=render(T[i]); gt=f['object'][i]
27:  inter=(mask&gt).sum(); union=(mask|gt).sum(); iou=inter/union if union else 0
28:  print(i,'iou',round(iou,4),'mask',mask.sum(),'gt',gt.sum(),'prec',round(inter/max(1,mask.sum()),3),'rec',round(inter/max(1,gt.sum()),3))
30:   out=np.zeros((H,W,3),np.uint8); out[gt]=[0,0,255]; out[mask]=[0,255,0]; out[gt&mask]=[255,255,255]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/mask=(rast\\[0,:,:,3\\]>=0).cpu().numpy()/print('rast',rast.dtype,rast.shape,rast[0,:,:,3].min().item(),rast[0,:,:,3].max().item(),(rast[0,:,:,3]>=0).sum().item()); mask=(rast[0,:,:,3]>=0).cpu().numpy()/\" /work/scratch/render_test.py; sed -i \"s/for i in \\[0,1,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,525,530,534,535,540,550,570,600,650,677\\]:/for i in [0]:/\" /work/scratch/render_test.py; python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
cuda True
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 0.0 1769472
0 iou 0.0089 mask 1769472 gt 15708 prec 0.009 rec 1.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/zz,torch.ones_like(zz)/zz\\/10,torch.ones_like(zz)/\" /work/scratch/render_test.py; sed -i \"s/for i in \\[0\\]:/for i in [0,90,150,240,330,450,540,570,650]:/\" /work/scratch/render_test.py; python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
cuda True
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95691.0 1769472
0 iou 0.0089 mask 1769472 gt 15708 prec 0.009 rec 1.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95692.0 1769472
90 iou 0.0088 mask 1769472 gt 15498 prec 0.009 rec 1.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95692.0 1769472
150 iou 0.0094 mask 1769472 gt 16704 prec 0.009 rec 1.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95759.0 1769472
240 iou 0.0164 mask 1769472 gt 28934 prec 0.016 rec 1.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95446.0 1769472
330 iou 0.0128 mask 1769472 gt 22673 prec 0.013 rec 1.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95439.0 1769472
450 iou 0.0037 mask 1769472 gt 6469 prec 0.004 rec 1.0
rast torch.float32 torch.Size([1, 1152 …[+337 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/rast\\[0,:,:,3\\]>=0/rast[0,:,:,3]>0/g\" /work/scratch/render_test.py; sed -i \"s/for i in \\[0,90,150,240,330,450,540,570,650\\]:/for i in [0,1,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,525,530,534,535,540,550,570,600,650,677]:/\" /work/scratch/render_test.py; python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
cuda True
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95691.0 15349
0 iou 0.0 mask 15349 gt 15708 prec 0.0 rec 0.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95613.0 15444
1 iou 0.0 mask 15444 gt 15355 prec 0.0 rec 0.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95692.0 15138
90 iou 0.0 mask 15138 gt 15498 prec 0.0 rec 0.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95692.0 15103
120 iou 0.0 mask 15103 gt 15600 prec 0.0 rec 0.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95692.0 16357
150 iou 0.0 mask 16357 gt 16704 prec 0.0 rec 0.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95750.0 19548
180 iou 0.0 mask 19548 gt 18656 prec 0.0 rec 0.0
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95759.0 24805
210 iou 0.0 mask 24805 gt …[+2358 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
for i in [0,150,240,330,450,650]:
 im=cv2.imread(f'/work/scratch/vis/iou_{i}.png'); q=im[:,:,0]
 # white both, green pred, red gt; infer bboxes for colors
 for name,mask in [('green',(im[:,:,1]>im[:,:,2])&(im[:,:,1]>im[:,:,0])),('red',(im[:,:,2]>im[:,:,1])&(im[:,:,2]>im[:,:,0])),('white',(im.min(2)>200))]:
  ys,xs=np.where(mask); print(i,name,(xs.min(),ys.min(),xs.max(),ys.max()) if len(xs) else None,len(xs))
PY"
```
<details><summary>output</summary>

```
0 green (760, 245, 911, 505) 15349
0 red (759, 648, 906, 905) 15708
0 white None 0
150 green (719, 220, 884, 500) 16357
150 red (723, 661, 884, 925) 16704
150 white None 0
240 green (729, 99, 933, 482) 28949
240 red (728, 674, 928, 1050) 28934
240 white None 0
330 green (911, 138, 1101, 470) 25635
330 red (907, 696, 1098, 1010) 22673
330 white None 0
450 green (682, 187, 828, 474) 21307
450 red (781, 691, 822, 962) 6469
450 white None 0
650 green (420, 215, 590, 500) 16538
650 red (424, 655, 589, 937) 18422
650 white None 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/1-2\\*vv\\/H/2*vv\\/H-1/\" /work/scratch/render_test.py; python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
cuda True
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95691.0 15348
0 iou 0.865 mask 15348 gt 15708 prec 0.938 rec 0.917
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95613.0 15444
1 iou 0.8672 mask 15444 gt 15355 prec 0.926 rec 0.932
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95692.0 15138
90 iou 0.8696 mask 15138 gt 15498 prec 0.941 rec 0.919
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95692.0 15103
120 iou 0.8692 mask 15103 gt 15600 prec 0.945 rec 0.915
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95692.0 16355
150 iou 0.8798 mask 16355 gt 16704 prec 0.946 rec 0.926
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0.0 95750.0 19548
180 iou 0.8851 mask 19548 gt 18656 prec 0.918 rec 0.962
rast torch.float32 torch.Size([1, 1152, 1536, 4]) 0. …[+2537 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_flips.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, cv2
import nvdiffrast.torch as dr
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32); Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(np.float32)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
glctx=dr.RasterizeCudaContext(); Kt=torch.tensor(K,device='cuda'); Twc_i=np.linalg.inv(Twc)
center=(mesh.bounds[0]+mesh.bounds[1])/2
flips={'id':np.eye(4,dtype=np.float32),
       'z180':np.diag([-1,-1,1,1]).astype(np.float32),
       'x180':np.diag([1,-1,-1,1]).astype(np.float32),
       'y180':np.diag([-1,1,-1,1]).astype(np.float32)}
def render(Twm):
 M=torch.tensor(Twc_i@Twm,dtype=torch.float32,device='cuda'); v=V@M[:3,:3].T+M[:3,3]; zz=v[:,2].clamp_min(1e-5)
 u=Kt[0,0]*v[:,0]/zz+Kt[0,2]; vv=Kt[1,1]*v[:,1]/zz+Kt[1,2]
 pos=torch.stack([2*u/W-1,2*vv/H-1,zz/10,torch.ones_like(zz)],-1)[None]
 rast,_=dr.rasterize(glctx,pos,F,resolution=[H,W]); return (rast[0,:,:,3]>0).cpu().numpy()
with h5py.File('/work/inputs/masks.h5','r') as f:
 obs=f['object'][:]; per=f['person'][:]
 for i in [0,60,120,150,180,210,240,270,300,330,360,390,420,450,480,510,525,540,550,570,600,650,677]:
  gt=obs[i] & ~per[i]
  v …[+273 chars]
```
<details><summary>output</summary>

```
0 id:0.865 z180:0.377 x180:0.341 y180:0.233 gt 15708
60 id:0.879 z180:0.349 x180:0.338 y180:0.211 gt 15764
120 id:0.870 z180:0.348 x180:0.348 y180:0.211 gt 15600
150 id:0.891 z180:0.253 x180:0.351 y180:0.225 gt 16689
180 id:0.918 z180:0.274 x180:0.387 y180:0.286 gt 18654
210 id:0.880 z180:0.292 x180:0.351 y180:0.309 gt 24416
240 id:0.807 z180:0.234 x180:0.331 y180:0.283 gt 28861
270 id:0.783 z180:0.287 x180:0.415 y180:0.257 gt 23863
300 id:0.810 z180:0.381 x180:0.326 y180:0.227 gt 23886
330 id:0.742 z180:0.507 x180:0.353 y180:0.303 gt 22673
360 id:0.818 z180:0.483 x180:0.359 y180:0.308 gt 20917
390 id:0.863 z180:0.418 x180:0.326 y180:0.259 gt 20770
420 id:0.857 z180:0.400 x180:0.369 y180:0.285 gt 16039
450 id:0.583 z180:0.526 x180:0.585 y180:0.469 gt 6462
480 id:0.864 z180:0.372 x180:0.326 …[+459 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float)
Rs=R.from_matrix(T[:,:3,:3]); drot=(Rs[:-1].inv()*Rs[1:]).magnitude(); dp=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
for i in np.argsort(drot)[-30:]: print(i,round(drot[i],3),round(dp[i],4))
print('median',np.median(drot),np.percentile(drot,90))
PY"
```
<details><summary>output</summary>

```
430 0.054 0.0171
428 0.056 0.0192
550 0.056 0.0174
176 0.059 0.0227
426 0.06 0.0217
545 0.06 0.0223
551 0.06 0.0169
178 0.061 0.0127
170 0.061 0.0087
432 0.062 0.0139
171 0.062 0.0084
452 0.063 0.0475
539 0.063 0.0167
542 0.064 0.0237
424 0.066 0.0186
447 0.066 0.0094
543 0.067 0.0251
553 0.069 0.0203
540 0.07 0.025
541 0.07 0.0222
544 0.07 0.0434
546 0.071 0.0193
548 0.072 0.0213
547 0.072 0.0248
549 0.079 0.0243
536 0.083 0.022
174 0.088 0.0205
530 0.093 0.0148
534 0.108 0.0226
533 0.115 0.0382
median 0.007118454367028056 0.038113823680587695

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/depth/moge2/depth_report.json; echo '---'; cat /pipeline/pipeline_report.json"
```
<details><summary>output</summary>

```
{
  "backend": "moge2",
  "batchSize": 8,
  "cameraPolicy": "first_frame_export_per_frame_depth",
  "depthM": {
    "max": 7.727181434631348,
    "min": 1.5775458812713623
  },
  "deviceCount": 1,
  "devices": [
    "cuda"
  ],
  "elapsedSeconds": 243.364520705014,
  "firstFrameIntrinsicsPx": [
    [
      804.0242919921875,
      0.0,
      768.0
    ],
    [
      0.0,
      804.0242919921875,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "focalXPx": {
    "max": 824.08544921875,
    "mean": 809.0067138671875,
    "min": 794.0028076171875
  },
  "focalYPx": {
    "max": 824.0855102539062,
    "mean": 809.0067138671875,
    "min": 794.0028076171875
  },
  "fps": 30.0,
  "frames": 678,
  "framesPerSecond": 2.785944302957022,
  "height": 1152,
  "intrinsicsFile": {
  …[+7422 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 print('keys',list(f.keys()))
 def show(name,obj):
  if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype); 
 f.visititems(show)
PY"
```
<details><summary>output</summary>

```
keys ['aligned', 'alignment', 'frame_names', 'raw']
aligned/front_stereo_camera_left (678,) object
alignment/front_stereo_camera_left/scale (678,) float32
alignment/front_stereo_camera_left/shift (678,) float32
alignment/front_stereo_camera_left/valid_count (678,) int32
frame_names/front_stereo_camera_left (678,) object
raw/front_stereo_camera_left (678,) object

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
for p in ['/pipeline/depth/moge2/intrinsics.pkl','/pipeline/foundationpose/episode_000021.pkl','/pipeline/export/episode_000021/edex']:
 print('\\n---',p)
 try:
  with open(p,'rb') as f: x=pickle.load(f)
  print(type(x))
  if isinstance(x,dict):
   print(x.keys())
   for k,v in x.items():
    try: print(k,type(v),getattr(v,'shape',None))
    except: pass
  else: print(x)
 except Exception as e:
  print(e)
  print(open(p,'rb').read(500))
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
dict_keys(['frames', 'camera_id', 'fp_poses_cam', 'fp_candidate_count', 'fp_selected_candidate_index', 'fp_selected_iou', 'fp_selected_visibility', 'fp_retry_count', 'fp_reliable', 'fp_oracle_rotation_error_deg', 'fp_translation_threshold_m', 'fp_selected_translation_distance_m', 'gt_object_visibility_ratio', 'gt_object_visibility_valid', 'gt_object_rendered_pixels', 'gt_object_visible_pixels', 'obj_pose_world', 'obj_rot_init', 'obj_t_init', 'metadata', 'fp_tracking_state', 'fp_confirmed_pose_source', 'fp_pending_pose_source', 'fp_emitted_pose_source', 'fp_gap_length', 'fp_observed_visible_pixels', 'fp_rendered_visible_pixels', 'fp_rendered_visibility', 'fp_full_image_iou', 'fp_tolerant_precision', 'fp_tolerant_recall', 'fp_centroid_distance', 'fp_confirmation_count', 'fp_re …[+2426 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 def rec(g,prefix=''):
  for k,v in g.items():
   if isinstance(v,h5py.Group):
    print(prefix+k,'GROUP'); rec(v,prefix+k+'/')
   else: print(prefix+k,v.shape,v.dtype)
 rec(f)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 g=f['aligned/front_stereo_camera_left'];
 print(type(g),g.shape,g.dtype)
 a=f[g[0]]
 print('obj',a.shape,a.dtype,a.min(),a.max(),a[600,800])
 r=f[f['raw/front_stereo_camera_left'][0]]; print('raw',r.shape,r.dtype,r.min(),r.max())
 print('names',f[f['frame_names/front_stereo_camera_left'][0]][()][:100])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
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

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 g=f['aligned/front_stereo_camera_left']; x=g[0]; print(repr(x),type(x));
 if hasattr(x,'item'): x=x.item()
 print(type(x),repr(x)); a=f[x]; print(a.shape,a.dtype,a.min(),a.max())
PY"
```
<details><summary>output</summary>

```
array([137,  80,  78, ...,  66,  96, 130], dtype=uint8) <class 'numpy.ndarray'>
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
ValueError: can only convert an array of size 1 to a Python scalar

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for p in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left','frame_names/front_stereo_camera_left']:
  g=f[p]; x=g[0]; print(p,g.shape,g.dtype,x.shape,x.dtype,x[:50])
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (678,) object (543855,) uint8 [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0   0   0   4 128  16   0   0   0   0  76  58   5   5   0   1   0
   0  73  68  65  84 120 218 196 253  93 147  36 185 178]
raw/front_stereo_camera_left (678,) object (513904,) uint8 [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0   0   0   4 128  16   0   0   0   0  76  58   5   5   0   1   0
   0  73  68  65  84 120 218 204 253 217 146  36 201 178]
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'bytes' object has no attribute 'shape'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for p in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  x=f[p][0]; a=cv2.imdecode(x,cv2.IMREAD_UNCHANGED); print(p,a.shape,a.dtype,a.min(),a.max(),a[600,800])
 x=f['frame_names/front_stereo_camera_left'][0]; print(type(x),repr(x))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (1152, 1536) uint16 1739 7616 3818
raw/front_stereo_camera_left (1152, 1536) uint16 1649 7219 3619
<class 'bytes'> b'000000'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
for i in [0,120,240,330,450,540,600,677]: print(i,np.round(Tc[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=np.array(m.vertices)
plane=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); n=plane[:3]; d=plane[3]
# sample vertices all each 10
for i in [0,60,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,650,677]:
 w=v[::10]@T[i,:3,:3].T+T[i,:3,3]; sd=w@n+d
 print(i,'ycenter',round((T[i,:3,3]+T[i,:3,:3]@v.mean(0))[1],3),'minmax y',round(w[:,1].min(),3),round(w[:,1].max(),3),'sd min/max/p1/p5',*[round(float(x),3) for x in [sd.min(),sd.max(),np.percentile(sd,1),np.percentile(sd,5)]])
PY"
```
<details><summary>output</summary>

```
0 ycenter 0.699 minmax y 0.329 1.205 sd min/max/p1/p5 -0.003 0.882 0.014 0.084
60 ycenter 0.69 minmax y 0.322 1.197 sd min/max/p1/p5 0.005 0.89 0.023 0.094
120 ycenter 0.699 minmax y 0.334 1.208 sd min/max/p1/p5 -0.006 0.879 0.014 0.086
150 ycenter 0.7 minmax y 0.334 1.208 sd min/max/p1/p5 0.001 0.886 0.019 0.089
180 ycenter 0.712 minmax y 0.348 1.222 sd min/max/p1/p5 0.009 0.886 0.016 0.087
210 ycenter 0.713 minmax y 0.347 1.226 sd min/max/p1/p5 0.007 0.884 0.013 0.085
240 ycenter 0.693 minmax y 0.329 1.206 sd min/max/p1/p5 0.006 0.883 0.013 0.085
270 ycenter 0.675 minmax y 0.308 1.182 sd min/max/p1/p5 0.008 0.886 0.014 0.086
300 ycenter 0.679 minmax y 0.304 1.186 sd min/max/p1/p5 -0.009 0.871 -0.003 0.069
330 ycenter 0.677 minmax y 0.302 1.187 sd min/max/p1/p5 -0.009 0.869 -0.001 0.07
36 …[+878 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float); cp=z['contact_prob_right_left'];
from scipy.spatial.transform import Rotation as R
p=T[:,:3,3]; Rs=R.from_matrix(T[:,:3,:3]); dr=(Rs[:-1].inv()*Rs[1:]).magnitude(); dp=np.linalg.norm(np.diff(p,axis=0),axis=1)
with h5py.File('/work/inputs/masks.h5','r') as f: o=f['object'][:]; per=f['person'][:]
cent=[]; area=[]
for i in range(len(o)):
 ys,xs=np.where(o[i]); cent.append([xs.mean(),ys.mean()]); area.append(len(xs))
cent=np.array(cent); dc=np.linalg.norm(np.diff(cent,axis=0),axis=1)
for i in range(0,678,10): print(i,'dp',round(dp[max(0,i-1)] if i else 0,4),'dr',round(np.degrees(dr[max(0,i-1)]) if i else 0,1),'dc',round(dc[max(0,i-1)] if i else 0,2),'contact',np.round(cp[i],2),'area',area[i])
print('---contact starts')
for th in [.1,.3,.5,.8]:
 ix=np.where(cp.max(1)>th)[0]; print(th,ix.min() if len(ix) else None,ix.max() if len(ix) else None,len(ix))
PY"
```
<details><summary>output</summary>

```
0 dp 0 dr 0 dc 0 contact [0. 0.] area 15708
10 dp 0.0017 dr 0.2 dc 0.49 contact [0. 0.] area 15422
20 dp 0.0064 dr 0.3 dc 0.07 contact [0. 0.] area 15707
30 dp 0.005 dr 0.3 dc 0.15 contact [0. 0.] area 15740
40 dp 0.0102 dr 0.8 dc 0.12 contact [0. 0.] area 15758
50 dp 0.0042 dr 0.5 dc 0.18 contact [0. 0.] area 15712
60 dp 0.0022 dr 0.1 dc 0.16 contact [0. 0.] area 15764
70 dp 0.0012 dr 0.2 dc 0.1 contact [0. 0.] area 15770
80 dp 0.0051 dr 0.1 dc 0.07 contact [0. 0.] area 15720
90 dp 0.005 dr 0.3 dc 1.31 contact [0. 0.] area 15498
100 dp 0.0027 dr 0.3 dc 0.45 contact [0.53 0.04] area 15893
110 dp 0.002 dr 0.5 dc 0.18 contact [1. 1.] area 15817
120 dp 0.0043 dr 0.2 dc 0.03 contact [1. 1.] area 15600
130 dp 0.006 dr 1.2 dc 1.26 contact [1. 1.] area 15945
140 dp 0.0093 dr 0.5 dc 0.98 contact [ …[+3058 chars]
```
</details>

**shell** (exit 2)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
mat=m.visual.material
print(type(mat)); print(mat.__dict__.keys())
for k,v in mat.__dict__.items():
 print(k,type(v),getattr(v,'shape',None),getattr(v,'size',None))
 if hasattr(v,'save'):
  try: v.save('/work/scratch/vis/material_'+k+'.png')
  except: pass
PY
ls -lh /work/scratch/vis/material* 2>/dev/null"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.material.PBRMaterial'>
dict_keys(['_data'])
_data <class 'dict'> None None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mat=m.visual.material
print(mat._data.keys())
for k,v in mat._data.items(): print(k,type(v),getattr(v,'shape',None),getattr(v,'size',None),repr(v)[:200])
img=mat.baseColorTexture
print('base',type(img),img.size,img.mode)
img.save('/work/scratch/vis/texture_base.png')
PY
ls -lh /work/scratch/vis/texture_base.png"
```
<details><summary>output</summary>

```
dict_keys(['baseColorFactor', 'metallicFactor', 'roughnessFactor', 'baseColorTexture', 'doubleSided'])
baseColorFactor <class 'numpy.ndarray'> (4,) 4 array([255, 255, 255, 255], dtype=uint8)
metallicFactor <class 'float'> None None 0.0
roughnessFactor <class 'float'> None None 0.5
baseColorTexture <class 'PIL.Image.Image'> None (5007, 4510) <PIL.Image.Image image mode=RGB size=5007x4510 at 0x7E74F2F0E210>
doubleSided <class 'bool'> None None False
base <class 'PIL.Image.Image'> (5007, 4510) RGB
-rw-rw-r-- 1 hanzhou hanzhou 12M Oct 20 22:02 /work/scratch/vis/texture_base.png

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as f:
 for ix in [0,120,150,240,330,450,540,600,677]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,ix); ok,im=cap.read()
  ys,xs=f['object'][ix][:].nonzero(); y0=max(0,ys.min()-50); y1=min(1151,ys.max()+50); x0=max(0,xs.min()-50); x1=min(1535,xs.max()+150)
  cv2.imwrite(f'/work/scratch/vis/raw_{ix}.jpg',im[y0:y1,x0:x1])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_rgb.py <<'PY'
import numpy as np, torch, trimesh, json, cv2
import nvdiffrast.torch as dr
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32); Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(np.float32)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
tex=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255
# torch texture [1,3,H,W]
tx=torch.tensor(tex).permute(2,0,1)[None].cuda()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda'); UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')
glctx=dr.RasterizeCudaContext(); Kt=torch.tensor(K,device='cuda'); Twc_i=np.linalg.inv(Twc)
def render(Twm):
 M=torch.tensor(Twc_i@Twm,dtype=torch.float32,device='cuda'); v=V@M[:3,:3].T+M[:3,3]; zz=v[:,2].clamp_min(1e-5)
 u=Kt[0,0]*v[:,0]/zz+Kt[0,2]; vv=Kt[1,1]*v[:,1]/zz+Kt[1,2]
 pos=torch.stack([2*u/W-1,2*vv/H-1,zz/10,torch.ones_like(zz)],-1)[None]
 rast,_=dr.rasterize(glctx,pos,F,resolution=[H,W]); uv=dr.interpolate(UV[None],rast,F)[0]
 # test variants
 outs=[]
 for yy in [uv[...,1],1-uv[...,1]]:
  grid=torch.stack([uv[...,0]*2-1, yy*2-1],-1)[None]
  col=torch.nn.functional.grid_sample(tx,grid,align_corners=False,mode='bilinear')[0].permute(1,2,0)
  outs.append(col.cpu().numpy())
 mas …[+644 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_rgb.py", line 26, in <module>
    mask,(a,b)=render(T[i]); cap=cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,i); _,im=cap.read()
               ^^^^^^^^^^^^
  File "/work/scratch/render_rgb.py", line 21, in render
    col=torch.nn.functional.grid_sample(tx,grid,align_corners=False,mode='bilinear')[0].permute(1,2,0)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/functional.py", line 4910, in grid_sample
    return torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: grid_sampler(): …[+146 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/grid=torch.stack(\\[uv\\[...,0\\]\\*2-1, yy\\*2-1\\],-1)\\[None\\]/grid=torch.stack([uv[...,0]*2-1, yy*2-1],-1)/\" /work/scratch/render_rgb.py; python /work/scratch/render_rgb.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.signature(dr.antialias)); print(inspect.getsource(dr.antialias)[:3000])
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
/usr/bin/bash -c "cat > /work/scratch/opt_static.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, cv2, time
import nvdiffrast.torch as dr
torch.manual_seed(0); np.random.seed(0)
W,H=1536,1152; SC=4; w,h=W//SC,H//SC
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32)/SC; K[2,2]=1; Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz'); Tbase=z['T_world_mesh'].astype(np.float32)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); center=(mesh.bounds[0]+mesh.bounds[1])/2
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda'); ones=torch.ones((1,len(V),1),device='cuda')
glctx=dr.RasterizeCudaContext(); Kt=torch.tensor(K,device='cuda'); Twc_i=torch.tensor(np.linalg.inv(Twc),device='cuda'); cc=torch.tensor(center,device='cuda')

def rod(w):
 th=torch.linalg.norm(w)+1e-12; a=w/th; Kx=torch.zeros(3,3,device=w.device); Kx[0,1]=-a[2];Kx[0,2]=a[1];Kx[1,0]=a[2];Kx[1,2]=-a[0];Kx[2,0]=-a[1];Kx[2,1]=a[0]
 return torch.eye(3,device=w.device)+torch.sin(th)*Kx+(1-torch.cos(th))*(Kx@Kx)

def poses(T,dt,dw):
 B=T.shape[0]; Rd=torch.stack([rod(dw[i]) for i in range(B)]); cw=(T[:,:3,:3]@cc)+T[:,:3,3]
 R=Rd@T[:,:3,:3]; t=(Rd@(T[:,:3,3]-cw).unsqueeze(-1)).squeeze(-1)+cw+dt
 out=torch.zeros(B,4,4,device=T.device); out[:,:3,:3]=R;out[:,:3,3]=t;out[:,3,3]=1;return out

def render(T):
 B=len(T); M=Twc_i[None]@T;  …[+2216 chars]
```
<details><summary>output</summary>

```
ids [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
Traceback (most recent call last):
  File "/work/scratch/opt_static.py", line 56, in <module>
    t=time.time(); T,dt,dw,ious=optimize(ids); print('elapsed',time.time()-t); print('base',np.round(Tbase[ids,0:3,3],4));
                                ^^^^^^^^^^^^^
  File "/work/scratch/opt_static.py", line 45, in optimize
    opt.zero_grad(); T=poses(T0,dt,dw); soft,_=render(T)
                       ^^^^^^^^^^^^^^^
  File "/work/scratch/opt_static.py", line 16, in poses
    B=T.shape[0]; Rd=torch.stack([rod(dw[i]) for i in range(B)]); cw=(T[:,:3,:3]@cc)+T[:,:3,3]
                                                                      ~~~~~~~~~~^~~
RuntimeError: expected scalar type Double but found Floa …[+2 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/cc=torch.tensor(center,device='cuda')/cc=torch.tensor(center,dtype=torch.float32,device='cuda')/\" /work/scratch/opt_static.py; python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
ids [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
Traceback (most recent call last):
  File "/work/scratch/opt_static.py", line 56, in <module>
    t=time.time(); T,dt,dw,ious=optimize(ids); print('elapsed',time.time()-t); print('base',np.round(Tbase[ids,0:3,3],4));
                                ^^^^^^^^^^^^^
  File "/work/scratch/opt_static.py", line 45, in optimize
    opt.zero_grad(); T=poses(T0,dt,dw); soft,_=render(T)
                                               ^^^^^^^^^
  File "/work/scratch/opt_static.py", line 24, in render
    rast,_=dr.rasterize(glctx,pos,F,resolution=[h,w]); col=dr.interpolate(ones.expand(B,-1,-1),rast,F)[0]; soft=dr.antialias(col,rast,pos,F)[...,0]
                                                           ^^^^^^^^^ …[+751 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c 'sed -i "s/ones.expand(B,-1,-1)/ones.repeat(B,1,1)/" /work/scratch/opt_static.py; python /work/scratch/opt_static.py'
```
<details><summary>output</summary>

```
ids [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
it 0 loss 0.08348903805017471 reg 0.0 dt [ 0.00071431  0.00128575 -0.00071425] dw [-0.00099995 -0.00014277  0.00042847]
it 20 loss 0.0833704024553299 reg 1.7246156858163886e-07 dt [ 0.0113871   0.0015128  -0.00234662] dw [-0.00421689  0.00196157 -0.00282723]
it 40 loss 0.08521763980388641 reg 2.905851772538881e-07 dt [ 0.01635703 -0.00119592 -0.00187166] dw [-0.00657881 -0.00053271 -0.00100658]
it 60 loss 0.08677782863378525 reg 3.7372643646449433e-07 dt [ 0.02056546 -0.0022284  -0.00401213] dw [-0.00770643  0.00017691  0.00242312]
it 80 loss 0.08472279459238052 reg 5.117242949381762e-07 dt [ 0.02678289 -0.00294416 -0.00407133] dw [-0.00658549  0.00027942  0.00323147]
it 99 loss 0.08593873679637909 r …[+638 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/dt.cpu().numpy(),dw.cpu().numpy()/dt.detach().cpu().numpy(),dw.detach().cpu().numpy()/\" /work/scratch/opt_static.py; sed -i \"s/print('elapsed',time.time()-t); print('base'/print('elapsed',time.time()-t); print('base'/\" /work/scratch/opt_static.py; python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
ids [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
it 0 loss 0.08348903805017471 reg 0.0 dt [ 0.00071431  0.00128575 -0.00071425] dw [-0.00099995 -0.00014277  0.00042847]
it 20 loss 0.08411940932273865 reg 1.8475540741746954e-07 dt [ 0.01145283  0.00185847 -0.00245342] dw [-0.00498185  0.00146245 -0.00417692]
it 40 loss 0.0843944326043129 reg 3.278713052168314e-07 dt [ 0.0163926  -0.00073799 -0.00203753] dw [-0.00621663  0.00031312 -0.00411655]
it 60 loss 0.08611339330673218 reg 3.9725421174807707e-07 dt [ 0.02030843 -0.00180331 -0.0040462 ] dw [-0.00721367  0.00170739 -0.00230081]
it 80 loss 0.08447684347629547 reg 5.476227897815988e-07 dt [ 0.02667454 -0.00269299 -0.00383771] dw [-0.00782156  0.00182825 -0.00260718]
it 99 loss 0.08539646118879318 r …[+3125 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_static.py'
s=open(p).read()
s=s.replace('dt=torch.zeros(B,3,device=\\'cuda\\',requires_grad=True); dw=torch.zeros(B,3,device=\\'cuda\\',requires_grad=True)\\n opt=torch.optim.Adam([dt,dw],lr=lr)','dt=torch.zeros(1,3,device=\\'cuda\\',requires_grad=True); dw=torch.zeros(1,3,device=\\'cuda\\',requires_grad=True)\\n opt=torch.optim.Adam([dt,dw],lr=lr)')
s=s.replace(\"T=poses(T0,dt,dw); soft,_=render(T)\\n  inter\", \"T=poses(T0,dt.repeat(B,1),dw.repeat(B,1)); soft,_=render(T)\\n  inter\")
s=s.replace(\"T=poses(T0,dt,dw); soft,_=render(T); ious\", \"T=poses(T0,dt.repeat(B,1),dw.repeat(B,1)); soft,_=render(T); ious\")
s=s.replace(\"lr=0.003\", \"lr=0.001\")
s=s.replace(\"steps=100\", \"steps=150\")
open('/work/scratch/opt_static_shared.py','w').write(s)
PY
python /work/scratch/opt_static_shared.py"
```
<details><summary>output</summary>

```
ids [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
it 0 loss 0.08348903805017471 reg 0.0 dt [ 0.001  0.001 -0.001] dw [-0.001       0.00099997  0.00099999]
it 20 loss 0.0792422965168953 reg 7.742447394321061e-08 dt [0.00973826 0.0022817  0.00056146] dw [-0.00966556  0.00571623 -0.00413392]
it 40 loss 0.08391983062028885 reg 1.4000812598169432e-07 dt [ 0.01838481 -0.00147196 -0.00085008] dw [-0.00932836 -0.00134976 -0.00284649]
it 60 loss 0.08786878734827042 reg 3.016841390035552e-07 dt [ 0.02559458 -0.00610328 -0.0027393 ] dw [-0.00967667 -0.00259271  0.01205057]
it 80 loss 0.09005139768123627 reg 5.731827172894555e-07 dt [ 0.03431712 -0.0084018  -0.00479979] dw [-0.01070739 -0.00018026  0.01954384]
it 100 loss 0.08992669731378555 reg 8.7503815393574 …[+2084 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_static_shared.py'; s=open(p).read()
s=s.replace(\"T0=torch.tensor(Tbase[ids],device='cuda')\", \"T0=torch.tensor(Tbase[ids[0]:ids[0]+1],device='cuda').repeat(B,1,1)\")
s=s.replace('lr=0.001','lr=0.0003')
s=s.replace('steps=150','steps=200')
open('/work/scratch/opt_static_shared2.py','w').write(s)
PY
python /work/scratch/opt_static_shared2.py"
```
<details><summary>output</summary>

```
ids [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
Traceback (most recent call last):
  File "/work/scratch/opt_static_shared2.py", line 56, in <module>
    t=time.time(); T,dt,dw,ious=optimize(ids); print('elapsed',time.time()-t); print('base',np.round(Tbase[ids,0:3,3],4));
                                ^^^^^^^^^^^^^
  File "/work/scratch/opt_static_shared2.py", line 41, in optimize
    T0=torch.tensor(Tbase[ids[0]:ids[0]+1],device='cuda').repeat(B,1,1); gt=get_masks(ids); B=len(ids)
                                                                 ^
UnboundLocalError: cannot access local variable 'B' where it is not associated with a value

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/T0=torch.tensor(Tbase\\[ids\\[0\\]:ids\\[0\\]+1\\],device='cuda').repeat(B,1,1); gt=get_masks(ids); B=len(ids)/B=len(ids); T0=torch.tensor(Tbase[ids[0]:ids[0]+1],device='cuda').repeat(B,1,1); gt=get_masks(ids)/\" /work/scratch/opt_static_shared2.py; python /work/scratch/opt_static_shared2.py"
```
<details><summary>output</summary>

```
ids [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
it 0 loss 0.09080153703689575 reg 0.0 dt [-0.0003 -0.0003 -0.0003] dw [-0.0003  0.0003  0.0003]
it 20 loss 0.07681917399168015 reg 1.821795692080741e-08 dt [-0.00017003  0.00217147 -0.00511916] dw [-0.0041769  -0.00253825  0.00230722]
it 40 loss 0.07800807058811188 reg 3.494703548767575e-08 dt [ 9.2314767e-05  3.1790778e-03 -7.8737577e-03] dw [-0.0044769  -0.00265743  0.00172621]
it 60 loss 0.07429303228855133 reg 2.1679291251075483e-08 dt [ 0.00041226  0.00063172 -0.00739857] dw [-0.00244083 -0.00142149 -0.00134434]
it 80 loss 0.07489576190710068 reg 2.1695543139799156e-08 dt [ 0.00080012  0.00036452 -0.00730563] dw [-0.00292944 -0.00053874  0.00135944]
it 100 loss 0.07451710104942322 reg 2.31903509 …[+2117 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_smooth.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, cv2, time
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
import nvdiffrast.torch as dr
W,H=1536,1152; SC=4; w,h=W//SC,H//SC
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32)/SC; K[2,2]=1; Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(np.float32); Fn=len(T)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda'); ones=torch.ones((1,len(V),1),device='cuda')
glctx=dr.RasterizeCudaContext(); Kt=torch.tensor(K,device='cuda'); Twc_i=torch.tensor(np.linalg.inv(Twc),device='cuda')
def render(Tin):
 M=torch.tensor(Twc_i[None].cpu().numpy()@Tin,dtype=torch.float32,device='cuda'); B=len(Tin); v=torch.einsum('bij,bvj->bvi',M[:,:3,:3],V[None])+M[:,None,:3,3]; zz=v[...,2].clamp_min(1e-5)
 u=Kt[0,0]*v[...,0]/zz+Kt[0,2]; vv=Kt[1,1]*v[...,1]/zz+Kt[1,2]
 pos=torch.stack([2*u/w-1,2*vv/h-1,zz/10,torch.ones_like(zz)],-1)
 rast,_=dr.rasterize(glctx,pos,F,resolution=[h,w]); return (rast[...,3]>0).cpu().numpy()
with h5py.File('/work/inputs/masks.h5','r') as f:
 obs=f['object'][:].astype(np.uint8); per=f['person'][:]
 # smaller visible masks
 gt=[]
 for i in range(Fn):
  pp=cv2.dila …[+1749 chars]
```
<details><summary>output</summary>

```
base all 0.711726123719609 static1 0.8627252349189941 manip 0.6604074150453202 static2 0.7709231940797231
win 5 poly 1 all 0.71257 manip 0.66163 delta 0.001221 betterframes 245
win 5 poly 2 all 0.71199 manip 0.6608 delta 0.000388 betterframes 236
win 5 poly 3 all 0.71199 manip 0.66079 delta 0.00038 betterframes 235
win 7 poly 1 all 0.71252 manip 0.66156 delta 0.001148 betterframes 237
win 7 poly 2 all 0.71209 manip 0.66093 delta 0.000523 betterframes 248
win 7 poly 3 all 0.71208 manip 0.66092 delta 0.000509 betterframes 248
win 9 poly 1 all 0.71203 manip 0.66085 delta 0.00044 betterframes 235
win 9 poly 2 all 0.71214 manip 0.66101 delta 0.000602 betterframes 239
win 9 poly 3 all 0.71213 manip 0.66099 delta 0.000585 betterframes 240
win 11 poly 1 all 0.71068 manip 0.65889 delta -0.001516 be …[+590 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_static_shared2.py'; s=open(p).read()
s=s.replace(\"ids=list(range(0,126,6)); print('ids',ids)\",\"ids=list(range(592,678,3)); print('ids',ids)\")
s=s.replace(\"np.savez('/work/scratch/static_init.npz'\",\"np.savez('/work/scratch/static_final.npz'\")
open('/work/scratch/opt_static_final.py','w').write(s)
PY
python /work/scratch/opt_static_final.py"
```
<details><summary>output</summary>

```
ids [592, 595, 598, 601, 604, 607, 610, 613, 616, 619, 622, 625, 628, 631, 634, 637, 640, 643, 646, 649, 652, 655, 658, 661, 664, 667, 670, 673, 676]
it 0 loss 0.13322792947292328 reg 0.0 dt [0.0003 0.0003 0.0003] dw [-0.0003 -0.0003  0.0003]
it 20 loss 0.10798399150371552 reg 4.122545860241189e-08 dt [-0.00280052  0.00501889  0.00557357] dw [-0.00321426 -0.00544813  0.00526596]
it 40 loss 0.10530208796262741 reg 7.247258793086075e-08 dt [-0.00408674  0.00471178  0.00693653] dw [-0.00857114 -0.00541916  0.00565941]
it 60 loss 0.10544802993535995 reg 9.739990503021545e-08 dt [-0.00469483  0.00454529  0.00748738] dw [-0.01057361 -0.00446524  0.00792975]
it 80 loss 0.10513344407081604 reg 1.0887521284530521e-07 dt [-0.00450019  0.00482138  0.00798007] dw [-0.01109728 -0.00317135  0.00926332]
 …[+2417 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); Tw=z['T_world_mesh'].astype(float); fp=z['T_cam_mesh_foundationpose'].astype(float); cc=z['T_cam_mesh_coconet'].astype(float)
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']); tc=np.linalg.inv(Twc)@Tw
for i in [0,60,120,150,180,240,300,360,420,450,480,520,540,570,600,650,677]:
 print(i,'fp',round(np.degrees((R.from_matrix(fp[i,:3,:3]).inv()*R.from_matrix(tc[i,:3,:3])).magnitude()),2),'cc',round(np.degrees((R.from_matrix(cc[i,:3,:3]).inv()*R.from_matrix(tc[i,:3,:3])).magnitude()),2),'fpcc',round(np.degrees((R.from_matrix(fp[i,:3,:3]).inv()*R.from_matrix(cc[i,:3,:3])).magnitude()),2))
PY"
```
<details><summary>output</summary>

```
0 fp 2.18 cc 1.0 fpcc 1.3
60 fp 1.03 cc 1.01 fpcc 0.79
120 fp 2.96 cc 1.01 fpcc 1.99
150 fp 1.65 cc 1.03 fpcc 0.63
180 fp 2.42 cc 1.08 fpcc 1.91
240 fp 2.95 cc 0.92 fpcc 2.4
300 fp 2.13 cc 0.75 fpcc 1.61
360 fp 3.64 cc 0.75 fpcc 3.0
420 fp 2.89 cc 0.91 fpcc 2.09
450 fp 2.57 cc 1.03 fpcc 1.75
480 fp 5.77 cc 1.02 fpcc 4.77
520 fp 4.97 cc 1.03 fpcc 4.51
540 fp 6.17 cc 1.11 fpcc 5.3
570 fp 4.52 cc 1.22 fpcc 4.19
600 fp 3.35 cc 1.24 fpcc 3.3
650 fp 3.54 cc 1.24 fpcc 3.28
677 fp 3.39 cc 1.24 fpcc 3.28

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
b=np.load('/work/scratch/base_iou.npy')
for i in range(0,678,10):
 print(i,round(b[i],3), 'avg10',round(b[i:i+10].mean(),3))
print('worst')
for i in np.argsort(b)[:50]: print(i,round(b[i],3))
PY"
```
<details><summary>output</summary>

```
0 0.85 avg10 0.852
10 0.865 avg10 0.859
20 0.859 avg10 0.86
30 0.857 avg10 0.853
40 0.838 avg10 0.855
50 0.879 avg10 0.863
60 0.869 avg10 0.868
70 0.87 avg10 0.871
80 0.878 avg10 0.879
90 0.865 avg10 0.859
100 0.863 avg10 0.866
110 0.864 avg10 0.867
120 0.874 avg10 0.872
130 0.864 avg10 0.868
140 0.866 avg10 0.874
150 0.879 avg10 0.885
160 0.853 avg10 0.843
170 0.858 avg10 0.834
180 0.832 avg10 0.846
190 0.852 avg10 0.845
200 0.845 avg10 0.834
210 0.822 avg10 0.831
220 0.859 avg10 0.839
230 0.854 avg10 0.822
240 0.775 avg10 0.778
250 0.779 avg10 0.76
260 0.764 avg10 0.797
270 0.766 avg10 0.688
280 0.686 avg10 0.717
290 0.737 avg10 0.751
300 0.765 avg10 0.744
310 0.743 avg10 0.741
320 0.76 avg10 0.717
330 0.65 avg10 0.613
340 0.539 avg10 0.56
350 0.558 avg10 0.597
360 0.662 avg10 0.681
370  …[+1173 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -vf \"select='between(n,400,565)*not(mod(n-400,5))',crop=800:800:250:300,scale=320:-1,tile=6x6\" -vsync vfr /work/scratch/vis/contact400.jpg && ls -lh /work/scratch/vis/contact400.jpg"
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 321K Oct 20 22:06 /work/scratch/vis/contact400.jpg

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/overlay_seq.py <<'PY'
import numpy as np,torch,trimesh,json,cv2
import nvdiffrast.torch as dr
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32); Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(np.float32)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
Kt=torch.tensor(K,device='cuda'); Twc_i=np.linalg.inv(Twc); glctx=dr.RasterizeCudaContext()
def mask(i):
 M=torch.tensor(Twc_i@T[i],dtype=torch.float32,device='cuda');v=V@M[:3,:3].T+M[:3,3];zz=v[:,2].clamp_min(1e-5)
 u=Kt[0,0]*v[:,0]/zz+Kt[0,2];vv=Kt[1,1]*v[:,1]/zz+Kt[1,2];pos=torch.stack([2*u/W-1,2*vv/H-1,zz/10,torch.ones_like(zz)],-1)[None]
 rast,_=dr.rasterize(glctx,pos,F,resolution=[H,W]);return (rast[0,:,:,3]>0).cpu().numpy().astype(np.uint8)
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
for i in range(400,571,5):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); _,im=cap.read(); m=mask(i); cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im,cnt,-1,(255,0,0),3); cv2.putText(im,str(i),(30,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),4)
 crop=im[300:1100,250:1150]; crop=cv2.resize(crop,(300,267)); frames.append(crop)
while len(frames)<36:frames.append(np.zeros_like …[+186 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/check_depth.py <<'PY'
import numpy as np,torch,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32); Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(np.float32)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
Kt=torch.tensor(K,device='cuda');Twci=np.linalg.inv(Twc);glctx=dr.RasterizeCudaContext()
def render(i):
 M=torch.tensor(Twci@T[i],dtype=torch.float32,device='cuda');v=V@M[:3,:3].T+M[:3,3];zz=v[:,2].clamp_min(1e-5)
 u=Kt[0,0]*v[:,0]/zz+Kt[0,2];vv=Kt[1,1]*v[:,1]/zz+Kt[1,2];pos=torch.stack([2*u/W-1,2*vv/H-1,zz/10,torch.ones_like(zz)],-1)[None]
 rast,_=dr.rasterize(glctx,pos,F,resolution=[H,W]);m=(rast[0,:,:,3]>0).cpu().numpy(); rz=(rast[0,:,:,2]*10).cpu().numpy();return m,rz
with h5py.File('/work/inputs/masks.h5','r') as fm, h5py.File('/pipeline/depth/moge2/aligned.h5','r') as fd:
 for i in [0,60,120,150,180,240,300,360,420,450,480,510,540,570,600,650,677]:
  o=fm['object'][i];p=fm['person'][i];m,rz=render(i); d=cv2.imdecode(fd['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
  # only valid object mask and visible, compare within rendered & object
  sel=o & m & ~p
  # on …[+419 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/check_depth.py", line 19, in <module>
    print(i,'MoGe obj med/mean',round(np.median(d[o&~p]),3),round(d[o&~p].mean(),3),'rendered',round(np.median(rz[sel2]),3),round(rz[sel2].mean(),3),'diff med',round(np.median(d[sel2]-rz[sel2]),3),'MAE',round(np.mean(abs(d[sel2]-rz[sel2])),3),'camera t',np.round(np.linalg.inv(Twc)@T[i,:3,3],2),'n',sel2.sum())
                                                                                                                                                                                                         ~~~~~~~~~~~~~~~~~~^~~~~~~~~~
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 4)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/np.round(np.linalg.inv(Twc)@T\\[i,:3,3\\],2)/np.round((np.linalg.inv(Twc)@T[i])[:3,3],2)/\" /work/scratch/check_depth.py; python /work/scratch/check_depth.py"
```
<details><summary>output</summary>

```
0 MoGe obj med/mean 2.826 2.896 rendered 2.889 2.934 diff med -0.091 MAE 0.111 camera t [0.26 0.65 3.04] n 14404
60 MoGe obj med/mean 2.959 3.032 rendered 2.859 2.904 diff med 0.074 MAE 0.111 camera t [0.26 0.65 3.02] n 14649
120 MoGe obj med/mean 2.855 2.928 rendered 2.884 2.929 diff med -0.067 MAE 0.09 camera t [0.25 0.65 3.04] n 14277
150 MoGe obj med/mean 2.722 2.801 rendered 2.727 2.773 diff med -0.029 MAE 0.067 camera t [0.11 0.66 2.87] n 15458
180 MoGe obj med/mean 2.558 2.607 rendered 2.563 2.605 diff med -0.02 MAE 0.055 camera t [-0.14  0.67  2.67] n 17936
240 MoGe obj med/mean 2.345 2.352 rendered 2.272 2.27 diff med 0.055 MAE 0.065 camera t [0.17 0.69 2.34] n 25444
300 MoGe obj med/mean 2.258 2.311 rendered 2.254 2.281 diff med -0.015 MAE 0.06 camera t [0.68 0.7  2.4 ] n 21446
3 …[+1149 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);
for i in [0,60,120,150,240,330,450,540,600,677]:
 print(i,'local z dot -n',np.round(T[i,:3,2]@(-n),4),'angles deg',np.degrees(np.arccos(np.clip(T[i,:3,2]@(-n),-1,1))))
PY"
```
<details><summary>output</summary>

```
0 local z dot -n 0.9986 angles deg 3.033891953491531
60 local z dot -n 0.9983 angles deg 3.342658703857415
120 local z dot -n 0.9972 angles deg 4.288153032056834
150 local z dot -n 0.9985 angles deg 3.121379386391185
240 local z dot -n 0.9992 angles deg 2.3548246062382217
330 local z dot -n 0.9994 angles deg 2.0037018498167076
450 local z dot -n 0.9995 angles deg 1.7721223372801103
540 local z dot -n 0.9994 angles deg 1.9312653958690071
600 local z dot -n 0.9997 angles deg 1.3040497236883108
677 local z dot -n 0.9997 angles deg 1.3528493722867285

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_static.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, cv2, time, argparse
import nvdiffrast.torch as dr
ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['init','final']); ap.add_argument('--src'); ap.add_argument('--sc',type=int,default=2); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--lr',type=float,default=1e-4); args=ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
W,H=1536,1152; SC=args.sc; w,h=W//SC,H//SC
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32)/SC; K[2,2]=1; Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz'); Tbase=z['T_world_mesh'].astype(np.float32)
if args.src: Tstart=np.load(args.src)['T'][0].astype(np.float32)
else: Tstart=Tbase[0]
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); center=(mesh.bounds[0]+mesh.bounds[1])/2
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda'); ones=torch.ones((1,len(V),1),device='cuda')
glctx=dr.RasterizeCudaContext(); Kt=torch.tensor(K,device='cuda'); Twc_i=torch.tensor(np.linalg.inv(Twc),device='cuda'); cc=torch.tensor(center,dtype=torch.float32,device='cuda')
def rod(w):
 th=torch.linalg.norm(w)+1e-12; a=w/th; Kx=torch.zeros(3,3,device=w.device); Kx[0,1]=-a[2];Kx[0,2]=a[1];Kx[1,0]=a[2];Kx[1,2]=-a[0];Kx[2,0]=-a[1];Kx[2,1]=a[0]
 retur …[+2427 chars]
```
<details><summary>output</summary>

```
phase init ids [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128] sc 2
it 0 loss 0.0722026377916336 dt [[ 9.999994e-05  9.999998e-05 -1.000000e-04]] dwdeg [[-0.00572916 -0.00572935 -0.00572953]]
it 20 loss 0.072012759745121 dt [[ 0.00138376  0.00044482 -0.00036226]] dwdeg [[-0.02449271 -0.07238498 -0.01665698]]
it 40 loss 0.07252187281847 dt [[ 2.405066e-03  8.007544e-05 -5.821431e-04]] dwdeg [[-0.01453357 -0.07126747  0.01768353]]
it 60 loss 0.07214773446321487 dt [[ 0.00319345 -0.00012973 -0.00079858]] dwdeg [[-0.01297467 -0.06448549 -0.01369304]]
it 80 loss 0.07218076288700104 dt [[ 0.00417148 -0.00071397 -0.00092958]] dwdeg [[ 0.02765019 -0.07136412 -0.02982242]]
it 99 loss 0.0725201889872551 dt [[ 0.00490684 -0.00087313 -0.00156885]] dwdeg [[ 0.04846195 -0.02669223 …[+285 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_static.py';s=open(p).read()
s=s.replace(\"ids=list(range(0,132,8))\",\"ids=list(range(0,132,16))\")
s=s.replace(\"ids=list(range(592,678,6))\",\"ids=list(range(592,678,11))\")
open('/work/scratch/refine_static_full.py','w').write(s)
PY
python /work/scratch/refine_static_full.py --phase init --src /work/scratch/init_sc2.npz --sc 1 --steps 80 --lr 5e-5"
```
<details><summary>output</summary>

```
phase init ids [0, 16, 32, 48, 64, 80, 96, 112, 128] sc 1
it 0 loss 0.0753374844789505 dt [[ 4.9999962e-05  4.9999995e-05 -4.9999991e-05]] dwdeg [[0.00286479 0.00286479 0.00286478]]
it 20 loss 0.0764000415802002 dt [[-3.815653e-05  5.736023e-04 -6.794748e-04]] dwdeg [[0.05475983 0.04491749 0.00046803]]
it 40 loss 0.07606483995914459 dt [[-0.00037658  0.00072399 -0.00072949]] dwdeg [[ 0.09743439  0.06157419 -0.00485895]]
it 60 loss 0.07606005668640137 dt [[-0.00064127  0.00090128 -0.00078833]] dwdeg [[ 0.11570815  0.07679816 -0.00517739]]
it 79 loss 0.07550807297229767 dt [[-0.00074666  0.00093851 -0.00047616]] dwdeg [[ 0.12042746  0.06970871 -0.00867836]]
sample first iou 0.8682489706950836
elapsed 2.1235153675079346 T [[-0.66386384  0.7471422  -0.03260696 -0.05608233]
 [-0.07883357 -0.026 …[+134 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/refine_static.py --phase final --src /work/scratch/static_final.npz --sc 2 --steps 100 --lr 5e-5'
```
<details><summary>output</summary>

```
phase final ids [592, 598, 604, 610, 616, 622, 628, 634, 640, 646, 652, 658, 664, 670, 676] sc 2
it 0 loss 0.09394460171461105 dt [[ 4.999998e-05 -4.999998e-05 -4.999999e-05]] dwdeg [[-0.00286479  0.00286479  0.00286479]]
it 20 loss 0.09190034121274948 dt [[ 0.00017413 -0.00054223 -0.00062054]] dwdeg [[-0.05797526  0.0500228   0.05430846]]
it 40 loss 0.0908391997218132 dt [[ 5.8453053e-05 -4.8195498e-04 -5.6113809e-04]] dwdeg [[-0.11111559  0.07200161  0.10622496]]
it 60 loss 0.08938653022050858 dt [[-0.00014523 -0.0005725  -0.00064315]] dwdeg [[-0.15979637  0.12461713  0.13111475]]
it 80 loss 0.08921210467815399 dt [[-0.0003543  -0.00055971 -0.00072774]] dwdeg [[-0.19571827  0.14755453  0.14200583]]
it 99 loss 0.08911773562431335 dt [[-0.00039856 -0.0004962  -0.00068542]] dwdeg [[-0.22321 …[+299 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/refine_static_full.py --phase final --src /work/scratch/final_sc2.npz --sc 1 --steps 80 --lr 5e-5'
```
<details><summary>output</summary>

```
phase final ids [592, 603, 614, 625, 636, 647, 658, 669] sc 1
it 0 loss 0.08797714114189148 dt [[ 4.9999933e-05 -4.9999984e-05 -4.9999991e-05]] dwdeg [[-0.00286479  0.00286479 -0.00286478]]
it 20 loss 0.08771063387393951 dt [[-0.00058321 -0.00018887 -0.00018435]] dwdeg [[-0.04782496  0.0178543  -0.04128113]]
it 40 loss 0.08724558353424072 dt [[-0.00094991  0.0001156  -0.00034976]] dwdeg [[-0.04520712 -0.0022885  -0.09102798]]
it 60 loss 0.08706759661436081 dt [[-0.00145124  0.00031159 -0.00046366]] dwdeg [[-0.04016296  0.01212307 -0.12506679]]
it 79 loss 0.08676962554454803 dt [[-0.0019153   0.00050736 -0.00060661]] dwdeg [[-0.04200559  0.0233353  -0.10574444]]
sample first iou 0.8411289465291804
elapsed 1.3456292152404785 T [[-0.1613821  -0.98679924  0.0135904   0.10142972]
 [ 0.02527751  …[+140 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_static_candidates.py <<'PY'
import numpy as np,torch,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(np.float32)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
Kt=torch.tensor(K,device='cuda');Twci=np.linalg.inv(Twc);glctx=dr.RasterizeCudaContext()
def render(Ti):
 M=torch.tensor(Twci[None]@Ti,dtype=torch.float32,device='cuda');v=torch.einsum('bij,bvj->bvi',M[:,:3,:3],V[None])+M[:,None,:3,3];zz=v[...,2].clamp_min(1e-5)
 u=Kt[0,0]*v[...,0]/zz+Kt[0,2];vv=Kt[1,1]*v[...,1]/zz+Kt[1,2];pos=torch.stack([2*u/W-1,2*vv/H-1,zz/10,torch.ones_like(zz)],-1);rast,_=dr.rasterize(glctx,pos,F,resolution=[H,W]);return (rast[...,3]>0).cpu().numpy()
def evaluate(ids,Ti,name):
 vals=[]
 with h5py.File('/work/inputs/masks.h5','r') as f:
  for s in range(0,len(ids),8):
   ii=ids[s:s+8];m=render(Ti[s:s+8]);o=f['object'][ii];p=f['person'][ii];
   for j in range(len(ii)):
    pp=cv2.dilate(p[j].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);g=o[j]&~pp;mm=m[j]&~pp;vals.append((mm&g).sum()/max(1,(mm|g).sum()))
 print(name,'mean',np.mean(vals),'median',np.median(vals), …[+499 chars]
```
<details><summary>output</summary>

```
base mean 0.871686481735022 median 0.8722945414844068 min 0.8449329884787209 max 0.8922852097667905
cand mean 0.8725147285603823 median 0.8794099878982835 min 0.5677237851662404 max 0.8924711027803811
delta 0.000828246825360379 better 0.7575757575757576 per-frame mean difference distribution [-0.31083964 -0.00539702  0.00047055  0.00853832  0.01555416  0.02007824
  0.03612588]
base mean 0.7649628723835931 median 0.7644117970371196 min 0.7243678387988938 max 0.8194697864119351
cand mean 0.8440939479860982 median 0.8413296863859281 min 0.838143566973253 max 0.8699740058625076
delta 0.07913107560250485 better 1.0 per-frame mean difference distribution [0.04924777 0.0687513  0.07334604 0.07826023 0.08276081 0.08882281
 0.11663546]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# rerun modified to find low; can't import due main executes. let's just inspect point where candidate maybe person overlap weird
# use full evaluator code by exec until for loop? quick exec
s=open('/work/scratch/eval_static_candidates.py').read().split(\"for phase,ids,cand\")[0]; ns={};exec(s,ns)
import numpy as np
ids=list(range(0,132));C=np.load('/work/scratch/init_sc1.npz')['T'][0];a=ns['evaluate'](ids,ns['T'][ids],'base');b=ns['evaluate'](ids,np.repeat(C[None],len(ids),0),'cand');d=b-a
for i in np.argsort(d)[:10]:print(i,a[i],b[i],d[i])
PY"
```
<details><summary>output</summary>

```
base mean 0.871686481735022 median 0.8722945414844068 min 0.8449329884787209 max 0.8922852097667905
cand mean 0.8725147285603823 median 0.8794099878982835 min 0.5677237851662404 max 0.8924711027803811
131 0.8785634258682933 0.5677237851662404 -0.31083964070205283
130 0.8856482638021483 0.6302543364903692 -0.2553939273117791
129 0.8851858747595109 0.6904761904761905 -0.19470968428332047
128 0.8753763903398267 0.7519086127919471 -0.12346777754787963
127 0.8755446455968089 0.7948658380764316 -0.08067880752037726
126 0.8848858729179518 0.8343645882912899 -0.050521284626661944
125 0.8835671466534825 0.8616660612664971 -0.02190108538698532
88 0.8922852097667905 0.8757072348621644 -0.01657797490462609
72 0.8877507727741075 0.877359059391016 -0.010391713383091528
77 0.8877823658003401 0.8780487804 …[+29 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_static_shared2.py';s=open(p).read();s=s.replace(\"ids=list(range(0,126,6)); print('ids',ids)\",\"ids=list(range(0,124,6)); print('ids',ids)\");s=s.replace(\"static_init.npz\",\"static_init_clean.npz\");open('/work/scratch/opt_static_init_clean.py','w').write(s)
PY
python /work/scratch/opt_static_init_clean.py > /work/scratch/opt_init_clean.log && tail -8 /work/scratch/opt_init_clean.log
# modify refine ids for init clean
python - <<'PY'
p='/work/scratch/refine_static.py';s=open(p).read();s=s.replace(\"ids=list(range(0,132,8))\",\"ids=list(range(0,124,8))\");open('/work/scratch/refine_static_clean.py','w').write(s)
PY
python /work/scratch/refine_static_clean.py --phase init --src /work/scratch/static_init_clean.npz --sc 2 --steps 100 --lr 2e-4"
```
<details><summary>output</summary>

```
 [-0.0602  0.6703  3.3531]
 [-0.0602  0.6703  3.3531]
 [-0.0602  0.6703  3.3531]
 [-0.0602  0.6703  3.3531]
 [-0.0602  0.6703  3.3531]
 [-0.0602  0.6703  3.3531]]
dt [[ 0.0013  0.0005 -0.0074]] dwdeg [[-0.13536729 -0.04474502  0.12206189]]
iou mean 0.884406194320211 range 0.8687068114511353 0.8981854838709677
phase init ids [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120] sc 2
it 0 loss 0.06659375876188278 dt [[ 0.0002  0.0002 -0.0002]] dwdeg [[ 0.0114591  -0.01145904 -0.01145911]]
it 20 loss 0.0683983862400055 dt [[ 0.00278998  0.00186092 -0.00101154]] dwdeg [[-0.04132964  0.14544797 -0.06163299]]
it 40 loss 0.06652729213237762 dt [[ 0.00398216  0.00232487 -0.00108877]] dwdeg [[ 0.02525189  0.25634757 -0.09232596]]
it 60 loss 0.06763856112957001 dt [[ 0.00405473  0.002058 …[+572 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_static_full.py';s=open(p).read();s=s.replace(\"ids=list(range(0,132,16))\",\"ids=list(range(0,124,15))\");open('/work/scratch/refine_static_full_clean.py','w').write(s)
PY
python /work/scratch/refine_static_full_clean.py --phase init --src /work/scratch/init_sc2.npz --sc 1 --steps 80 --lr 5e-5"
```
<details><summary>output</summary>

```
phase init ids [0, 15, 30, 45, 60, 75, 90, 105, 120] sc 1
it 0 loss 0.06519786268472672 dt [[ 4.9999573e-05 -4.9999991e-05  4.9999999e-05]] dwdeg [[ 0.00286479 -0.00286478 -0.00286479]]
it 20 loss 0.06275711953639984 dt [[-0.00060577 -0.00086872  0.00036351]] dwdeg [[ 0.03103366  0.02701446 -0.02711759]]
it 40 loss 0.06364688277244568 dt [[-0.00105417 -0.00103073  0.00019494]] dwdeg [[0.06300274 0.0263593  0.00842099]]
it 60 loss 0.06387891620397568 dt [[-0.00104186 -0.00101515  0.00026766]] dwdeg [[0.02533164 0.04657915 0.00989928]]
it 79 loss 0.06369078904390335 dt [[-0.00115082 -0.00099312  0.00031851]] dwdeg [[ 0.0206665   0.04445203 -0.00831958]]
sample first iou 0.8742394743246532
elapsed 1.625356912612915 T [[-0.66026884  0.75042963 -0.03000931 -0.05660216]
 [-0.07641724 -0.02737851 …[+129 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_static_candidates.py').read().split(\"for phase,ids,cand\")[0];ns={};exec(s,ns);import numpy as np
ids=list(range(0,125));C=np.load('/work/scratch/init_sc1.npz')['T'][0];a=ns['evaluate'](ids,ns['T'][ids],'base');b=ns['evaluate'](ids,np.repeat(C[None],len(ids),0),'cand');print('delta',np.mean(b-a),'better',np.mean(b>a),np.percentile(b-a,[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
base mean 0.871150751752679 median 0.87176414189837 min 0.8449329884787209 max 0.8922852097667905
cand mean 0.885904349198902 median 0.8849095544494354 min 0.8767436267436267 max 0.8989201406328479
delta 0.014753597446223144 better 0.952 [-0.01090365  0.00280646  0.0150352   0.02602342  0.0417357 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_static_full_clean.py';s=open(p).read();s=s.replace(\"ids=list(range(0,124,15))\",\"ids=list(range(0,124,8))\");open('/work/scratch/refine_static_full_clean16.py','w').write(s)
PY
python /work/scratch/refine_static_full_clean16.py --phase init --src /work/scratch/init_sc2.npz --sc 1 --steps 60 --lr 5e-5"
```
<details><summary>output</summary>

```
phase init ids [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120] sc 1
it 0 loss 0.06390076875686646 dt [[ 4.9999857e-05 -4.9999981e-05  4.9999995e-05]] dwdeg [[ 0.00286479 -0.00286478 -0.00286479]]
it 20 loss 0.06207245960831642 dt [[-0.00064445 -0.00095784  0.00024473]] dwdeg [[ 0.04194212  0.02748212 -0.02500102]]
it 40 loss 0.06205010041594505 dt [[-0.00095491 -0.00114309  0.00028467]] dwdeg [[ 0.06447786  0.04716596 -0.00082173]]
it 59 loss 0.06232992187142372 dt [[-0.0009462  -0.00108363  0.00024648]] dwdeg [[0.0458191  0.05798417 0.00538753]]
sample first iou 0.873980771571133
elapsed 1.7683115005493164 T [[-0.660074    0.75059205 -0.03022976 -0.05638311]
 [-0.07690319 -0.02748899  0.9966596   0.6703632 ]
 [ 0.7472537   0.6601938   0.07586768  3.3513362 ]
 [ 0.        …[+40 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_static_candidates.py').read().split(\"for phase,ids,cand\")[0];ns={};exec(s,ns);import numpy as np
ids=list(range(0,125));C=np.load('/work/scratch/init_sc1.npz')['T'][0];a=ns['evaluate'](ids,ns['T'][ids],'base');b=ns['evaluate'](ids,np.repeat(C[None],len(ids),0),'cand');print('delta',np.mean(b-a),'better',np.mean(b>a),np.percentile(b-a,[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
base mean 0.871150751752679 median 0.87176414189837 min 0.8449329884787209 max 0.8922852097667905
cand mean 0.8859227026928129 median 0.885016998542982 min 0.8764281419122069 max 0.8988192438135912
delta 0.014771950940133888 better 0.952 [-0.01110668  0.00284374  0.01533638  0.02602525  0.04164741]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_static_full_clean16.py';s=open(p).read();s=s.replace(\"ids=list(range(0,124,8))\",\"ids=list(range(592,678,8))\");open('/work/scratch/refine_static_final16.py','w').write(s)
PY
python /work/scratch/refine_static_final16.py --phase final --src /work/scratch/final_sc2.npz --sc 1 --steps 60 --lr 5e-5"
```
<details><summary>output</summary>

```
phase final ids [592, 603, 614, 625, 636, 647, 658, 669] sc 1
it 0 loss 0.08797714114189148 dt [[ 4.9999937e-05 -4.9999984e-05 -4.9999991e-05]] dwdeg [[-0.00286479  0.00286479 -0.00286478]]
it 20 loss 0.08771063387393951 dt [[-0.00058321 -0.00018887 -0.00018435]] dwdeg [[-0.04782496  0.0178543  -0.04128113]]
it 40 loss 0.08724558353424072 dt [[-0.00094991  0.0001156  -0.00034976]] dwdeg [[-0.04520712 -0.0022885  -0.09102798]]
it 59 loss 0.08705470710992813 dt [[-0.00142253  0.00030527 -0.00046035]] dwdeg [[-0.04001968  0.01027138 -0.1259716 ]]
sample first iou 0.840729127916246
elapsed 1.2084071636199951 T [[-0.16114806 -0.9868327   0.01393788  0.10188314]
 [ 0.02536835  0.00997604  0.9996235   0.73682904]
 [-0.9866044   0.16143918  0.02342644  2.2757988 ]
 [ 0.          0.          0.     …[+19 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'grep ids= /work/scratch/refine_static_final16.py'
```
<details><summary>output</summary>

```
if args.phase=='init': ids=list(range(592,678,8))
else: ids=list(range(592,678,11))
np.savez('/work/scratch/'+args.phase+f'_sc{SC}.npz',T=T.cpu().numpy(),ids=ids,dt=dt.detach().cpu().numpy(),dw=dw.detach().cpu().numpy())

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed -i "s/ids=list(range(592,678,11))/ids=list(range(592,678,8))/" /work/scratch/refine_static_final16.py; python /work/scratch/refine_static_final16.py --phase final --src /work/scratch/final_sc2.npz --sc 1 --steps 60 --lr 5e-5'
```
<details><summary>output</summary>

```
phase final ids [592, 600, 608, 616, 624, 632, 640, 648, 656, 664, 672] sc 1
it 0 loss 0.08851679414510727 dt [[ 4.9999908e-05 -4.9999988e-05 -4.9999995e-05]] dwdeg [[-0.00286479  0.00286479 -0.00286479]]
it 20 loss 0.0874335989356041 dt [[-0.00039531 -0.0004294  -0.00040993]] dwdeg [[-0.05764309  0.04194382 -0.03276384]]
it 40 loss 0.0874895378947258 dt [[-0.00102634 -0.00040926 -0.00052302]] dwdeg [[-0.09274378  0.04687549 -0.0083774 ]]
it 59 loss 0.08741595596075058 dt [[-0.00170503 -0.00035044 -0.00057248]] dwdeg [[-0.11244451  0.08198556  0.04451728]]
sample first iou 0.8405589204122835
elapsed 1.3122575283050537 T [[-0.16245814 -0.98665506  0.0109922   0.10191279]
 [ 0.02364155  0.0072448   0.99968934  0.73621637]
 [-0.9864324   0.16266577  0.02214883  2.275801  ]
 [ 0.          0.   …[+33 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_static_candidates.py').read().split(\"for phase,ids,cand\")[0];ns={};exec(s,ns);import numpy as np
ids=list(range(592,678));C=np.load('/work/scratch/final_sc1.npz')['T'][0];a=ns['evaluate'](ids,ns['T'][ids],'base');b=ns['evaluate'](ids,np.repeat(C[None],len(ids),0),'cand');print('delta',np.mean(b-a),'better',np.mean(b>a),np.percentile(b-a,[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
base mean 0.7649628723835931 median 0.7644117970371196 min 0.7243678387988938 max 0.8194697864119351
cand mean 0.8435319277103467 median 0.8407058522373452 min 0.8374438869817797 max 0.8694186560565871
delta 0.07856905532675372 better 1.0 [0.04869422 0.06825479 0.07755469 0.08816931 0.11602719]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_static_candidates.py').read().split(\"for phase,ids,cand\")[0];ns={};exec(s,ns);import numpy as np
ids=list(range(592,678));
for p in ['/work/scratch/static_final.npz','/work/scratch/final_sc2.npz','/work/scratch/final_sc1.npz']:
 C=np.load(p)['T'][0]; b=ns['evaluate'](ids,np.repeat(C[None],len(ids),0),p)
PY"
```
<details><summary>output</summary>

```
/work/scratch/static_final.npz mean 0.8322300206886701 median 0.8297799386966471 min 0.8265847381965233 max 0.8552227342549923
/work/scratch/final_sc2.npz mean 0.841907192039072 median 0.8391764175905682 min 0.8357681556879911 max 0.8674133745310086
/work/scratch/final_sc1.npz mean 0.8435319277103467 median 0.8407058522373452 min 0.8374438869817797 max 0.8694186560565871

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
Ci=np.load('/work/scratch/init_sc1.npz')['T'][0];Cf=np.load('/work/scratch/final_sc1.npz')['T'][0]
for name,C,idx in [('init',Ci,124),('final',Cf,592)]:
 dC=np.linalg.norm(T[idx,:3,3]-C[:3,3]);da=np.degrees((R.from_matrix(T[idx,:3,:3]).inv()*R.from_matrix(C[:3,:3])).magnitude())
 print(name,idx,'dt',dC,'dr',da,'base t',T[idx,:3,3],'cand',C[:3,3])
# candidate changes relative at boundaries
for i in [123,124,125,126,590,591,592,593]:
 C=Ci if i<125 else Cf if i>=592 else T[i]
 print(i, np.round(C[:3,3],4))
PY"
```
<details><summary>output</summary>

```
init 124 dt 0.009859753628736832 dr 1.5414247651742163 base t [-0.06190366  0.66940254  3.34322357] cand [-0.05638311  0.6703632   3.3513362 ]
final 592 dt 0.010328023334579887 dr 0.9379848656939296 base t [0.10601765 0.73220593 2.26721406] cand [0.10191279 0.73621637 2.275801  ]
123 [-0.0564  0.6704  3.3513]
124 [-0.0564  0.6704  3.3513]
125 [-0.0526  0.6675  3.3413]
126 [-0.0527  0.6706  3.3376]
590 [0.0998 0.7348 2.2653]
591 [0.1057 0.7341 2.2674]
592 [0.1019 0.7362 2.2758]
593 [0.1019 0.7362 2.2758]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for i in range(400,601,5): print(i,np.round(T[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
400 [0.615 0.652 3.576]
405 [0.609 0.649 3.52 ]
410 [0.613 0.654 3.438]
415 [0.623 0.648 3.356]
420 [0.612 0.646 3.288]
425 [0.602 0.67  3.214]
430 [0.556 0.684 3.123]
435 [0.537 0.682 3.06 ]
440 [0.497 0.68  3.043]
445 [0.395 0.678 3.052]
450 [0.312 0.69  3.07 ]
455 [0.237 0.679 3.091]
460 [0.156 0.673 3.131]
465 [0.082 0.668 3.185]
470 [0.023 0.67  3.23 ]
475 [-0.03   0.673  3.271]
480 [-0.063  0.68   3.296]
485 [-0.047  0.672  3.297]
490 [-0.058  0.669  3.299]
495 [-0.032  0.666  3.294]
500 [-0.026  0.665  3.293]
505 [-0.031  0.666  3.295]
510 [-0.033  0.671  3.295]
515 [-0.034  0.671  3.281]
520 [-0.017  0.677  3.231]
525 [-2.000e-03  6.830e-01  3.162e+00]
530 [0.058 0.676 3.081]
535 [0.035 0.689 2.973]
540 [0.078 0.698 2.88 ]
545 [0.129 0.693 2.771]
550 [0.142 0.702 2.668]
555 [0.165  …[+229 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);kp=z['human_keypoints_world'];cp=z['contact_prob_right_left'];
p=T[:,:3,3];v=np.diff(p,axis=0);sp=np.linalg.norm(v,axis=1)
for i in range(380,581,5):
 rh=kp[i,21:42].mean(0);lh=kp[i,42:63].mean(0)
 print(i,'obj',np.round(p[i],2),'sp',round(sp[max(0,i-1)],3),'R',np.round(rh,2),'L',np.round(lh,2),'cp',np.round(cp[i],2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);v=np.array(trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().vertices)[::5]
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=pl[:3];d=pl[3]
for i in range(0,678,15):
 w=v@T[i,:3,:3].T+T[i,:3,3]; sd=w@n+d
 print(i,round(np.percentile(sd,1),3),round(np.percentile(sd,5),3),round(np.percentile(sd,50),3),round((T[i,:3,3]+T[i,:3,:3]@v.mean(0))[1],3))
PY"
```
<details><summary>output</summary>

```
0 0.015 0.081 0.572 0.7
15 0.017 0.083 0.575 0.697
30 0.025 0.091 0.582 0.69
45 0.028 0.095 0.588 0.685
60 0.023 0.089 0.581 0.691
75 0.017 0.083 0.575 0.698
90 0.017 0.084 0.577 0.697
105 0.014 0.08 0.573 0.7
120 0.014 0.082 0.576 0.7
135 0.014 0.082 0.572 0.701
150 0.019 0.085 0.577 0.701
165 0.023 0.091 0.582 0.699
180 0.016 0.084 0.574 0.713
195 0.014 0.082 0.573 0.717
210 0.013 0.081 0.572 0.714
225 0.011 0.079 0.57 0.708
240 0.013 0.081 0.572 0.694
255 0.015 0.083 0.574 0.682
270 0.014 0.082 0.572 0.676
285 0.011 0.078 0.568 0.672
300 -0.002 0.065 0.555 0.679
315 0.004 0.072 0.562 0.673
330 -0.001 0.066 0.557 0.678
345 -0.001 0.067 0.557 0.678
360 -0.0 0.068 0.559 0.678
375 -0.004 0.063 0.554 0.681
390 -0.004 0.064 0.554 0.686
405 0.007 0.086 0.578 0.678
420 0.017 0.105 0.6 0.675
435 …[+466 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);v=np.array(trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().vertices)[::3]
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);d=json.load(open('/work/inputs/ground_plane.json'))['plane'][3]
mn=[]
for i in range(len(T)):
 w=v@T[i,:3,:3].T+T[i,:3,3];mn.append((w@n+d).min())
mn=np.array(mn);print('min dist frame stats',mn.min(),mn.max(),np.percentile(mn,[1,5,50,95,99]));print('above .02',np.where(mn>.02)[0][:100], 'count',np.sum(mn>.02));print('lowest idx',np.argsort(mn)[:20])
PY"
```
<details><summary>output</summary>

```
min dist frame stats -0.031194682780175675 0.01819239114094029 [-0.01474524 -0.01047173  0.00088951  0.01011942  0.01402655]
above .02 [] count 0
lowest idx [444 443 449 453 451 391 434 537 386 392 452 429 360 442 361 393 362 431
 390 326]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_trend.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, cv2, time
from scipy.spatial.transform import Rotation as Rot
from scipy.sparse import diags, eye, vstack
from scipy.sparse.linalg import spsolve
import nvdiffrast.torch as dr
W,H=1536,1152; SC=4; w,h=W//SC,H//SC
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],dtype=np.float32)/SC; K[2,2]=1; Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(np.float32); Fn=len(T)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
Kt=torch.tensor(K,device='cuda'); Twci=np.linalg.inv(Twc); glctx=dr.RasterizeCudaContext()
def render(Tin):
 M=torch.tensor(Twci[None]@Tin,dtype=torch.float32,device='cuda');B=len(Tin);v=torch.einsum('bij,bvj->bvi',M[:,:3,:3],V[None])+M[:,None,:3,3];zz=v[...,2].clamp_min(1e-5)
 u=Kt[0,0]*v[...,0]/zz+Kt[0,2];vv=Kt[1,1]*v[...,1]/zz+Kt[1,2];pos=torch.stack([2*u/w-1,2*vv/h-1,zz/10,torch.ones_like(zz)],-1);rast,_=dr.rasterize(glctx,pos,F,resolution=[h,w]);return (rast[...,3]>0).cpu().numpy()
# masks and pred confidence
with h5py.File('/work/inputs/masks.h5','r') as f:
 obs=f['object'][:].astype(np.uint8); per=f['person'][:]
gt=[];pd=[];conf=[]
for s in range(0,Fn,16):
 pp=np.array([cv2.dilate(per[i].astype(np.ui …[+1721 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_trend.py", line 21, in <module>
    g=np.array([cv2.resize((oo[j]&~pp[j]),(w,h),interpolation=cv2.INTER_AREA)>0.5 for j in range(len(oo))])
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/eval_trend.py", line 21, in <listcomp>
    g=np.array([cv2.resize((oo[j]&~pp[j]),(w,h),interpolation=cv2.INTER_AREA)>0.5 for j in range(len(oo))])
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'resize'
> Overload resolution failed:
>  - src data type = bool is not supported
>  - Expected Ptr<cv::UMat> for argument 'src'


```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/cv2.resize((oo\\[j\\]&~pp\\[j\\]),(w,h)/cv2.resize((oo[j]\\&~pp[j]).astype(np.uint8),(w,h)/\" /work/scratch/eval_trend.py; python /work/scratch/eval_trend.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_trend.py", line 25, in <module>
    pred=m[j]; occ=(pred&pp[j]).sum()/max(1,pred.sum()); conf.append(np.clip(1-occ,.03,1))
                    ~~~~^~~~~~
ValueError: operands could not be broadcast together with shapes (288,384) (1152,1536) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_trend.py';s=open(p).read();s=s.replace(\"pred=m[j]; occ=(pred&pp[j]).sum()/max(1,pred.sum())\",\"pred=m[j]; ppj=cv2.resize(pp[j].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool); occ=(pred&ppj).sum()/max(1,pred.sum())\");open(p,'w').write(s)
PY
python /work/scratch/eval_trend.py"
```
<details><summary>output</summary>

```
conf stats [0.14758497 0.31314647 0.80346106 0.94605346 1.         1.
 1.        ]
p 0 ltlr 1 1 all 0.71692 manip 0.66769 d 0.00046 better 245
p 0 ltlr 10 10 all 0.71711 manip 0.66797 d 0.000738 better 244
p 0 ltlr 100 100 all 0.71489 manip 0.66476 d -0.002475 better 212
p 0 ltlr 1000 1000 all 0.70306 manip 0.64758 d -0.019657 better 158
p 0 ltlr 100 1000 all 0.71025 manip 0.65802 d -0.009213 better 169
p 0 ltlr 1000 10000 all 0.6856 manip 0.62223 d -0.045 better 109
p 1 ltlr 1 1 all 0.71696 manip 0.66776 d 0.000524 better 248
p 1 ltlr 10 10 all 0.71705 manip 0.66788 d 0.000651 better 247
p 1 ltlr 100 100 all 0.71429 manip 0.66388 d -0.003351 better 210
p 1 ltlr 1000 1000 all 0.69978 manip 0.64281 d -0.024424 better 143
p 1 ltlr 100 1000 all 0.70839 manip 0.65532 d -0.011917 better 160
p 1 …[+459 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_static_all.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, cv2, time, argparse
import nvdiffrast.torch as dr
ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['init','final']);ap.add_argument('--src');ap.add_argument('--steps',type=int,default=40);ap.add_argument('--lr',type=float,default=3e-5);args=ap.parse_args()
torch.manual_seed(0);np.random.seed(0);W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load('/work/inputs/reconstruction.npz');mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();center=(mesh.bounds[0]+mesh.bounds[1])/2
T=np.load(args.src)['T'][0].astype(np.float32);ids=np.arange(0,125) if args.phase=='init' else np.arange(592,678)
# load all visible masks once
vis=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for s in range(0,len(ids),16):
  ii=ids[s:s+16];o=f['object'][ii];p=f['person'][ii]
  for j in range(len(ii)):
   pp=cv2.dilate(p[j].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);vis.append(o[j]&~pp)
vis=np.array(vis,dtype=np.uint8)
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');ones=torch.ones((1,len(V),1),device='cuda');Kt=torch.tensor(K,device='cuda');Twci=torch.tensor(np.linalg.inv(Twc),device='cuda');cc=torch.tensor(center,dtype=torch.float32,device=' …[+2067 chars]
```
<details><summary>output</summary>

```
it 0 loss 0.08567202294414694 dt [ 2.9999275e-05  2.9999988e-05 -2.9999985e-05] dwdeg [-0.00171886  0.00171887 -0.00171886]
it 10 loss 0.0856179032813419 dt [-0.0002227   0.00013768 -0.00011522] dwdeg [0.00262475 0.01748921 0.0082655 ]
it 20 loss 0.08557291193441911 dt [-0.0004046   0.00022549 -0.00014957] dwdeg [0.00863259 0.02445837 0.01465577]
it 30 loss 0.08553255349397659 dt [-0.00053351  0.0002566  -0.00022809] dwdeg [0.01899538 0.01783534 0.01410959]
it 39 loss 0.08528702841563658 dt [-0.00066499  0.00029068 -0.00022778] dwdeg [0.02072894 0.00953859 0.0181287 ]
elapsed 12.525859832763672 T [[-0.16262987 -0.9866302   0.01067961  0.10128196]
 [ 0.02394699  0.00687374  0.9996847   0.73649716]
 [-0.98639673  0.16283259  0.0225087   2.2755394 ]
 [ 0.          0.          0.          1.   …[+9 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_static_candidates.py').read().split(\"for phase,ids,cand\")[0];ns={};exec(s,ns);import numpy as np
for ph,ids,p in [('init',list(range(0,125)),'/work/scratch/init_sc1.npz'),('final',list(range(592,678)),'/work/scratch/final_all.npz')]:
 C=np.load(p)['T'][0];a=ns['evaluate'](ids,ns['T'][ids],'base '+ph);b=ns['evaluate'](ids,np.repeat(C[None],len(ids),0),'cand '+ph);print('delta',np.mean(b-a),'better',np.mean(b>a),np.percentile(b-a,[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
base init mean 0.871150751752679 median 0.87176414189837 min 0.8449329884787209 max 0.8922852097667905
cand init mean 0.8859227026928129 median 0.885016998542982 min 0.8764281419122069 max 0.8988192438135912
delta 0.014771950940133888 better 0.952 [-0.01110668  0.00284374  0.01533638  0.02602525  0.04164741]
base final mean 0.7649628723835931 median 0.7644117970371196 min 0.7243678387988938 max 0.8194697864119351
cand final mean 0.8443744045309528 median 0.841554439788512 min 0.8381063087815703 max 0.8703345313795964
delta 0.07941153214735941 better 1.0 [0.04960807 0.06910356 0.07841686 0.0889314  0.1168902 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/refine_static_all.py --phase init --src /work/scratch/init_sc1.npz --steps 40 --lr 3e-5'
```
<details><summary>output</summary>

```
it 0 loss 0.06291682971641421 dt [ 2.9999901e-05 -2.9999952e-05  2.9999985e-05] dwdeg [ 0.00171885 -0.00171887 -0.00171886]
it 10 loss 0.06283403793349862 dt [ 1.1241784e-04 -5.2288015e-05 -2.3462264e-06] dwdeg [ 0.00113227 -0.00320183 -0.00403301]
it 20 loss 0.06279776757583022 dt [ 1.787937e-04 -7.456350e-05 -4.648206e-06] dwdeg [ 0.001568   -0.00186814 -0.00464203]
it 30 loss 0.06315283058211207 dt [ 2.3914559e-04 -9.3820905e-05 -2.3747722e-05] dwdeg [ 0.00147055 -0.00108704 -0.00555112]
it 39 loss 0.06270919949747622 dt [ 2.85299524e-04 -1.09602006e-04 -3.85533640e-05] dwdeg [ 0.00104764 -0.00356083 -0.00499241]
elapsed 19.43216896057129 T [[-0.6601271   0.75054866 -0.03014763 -0.05610376]
 [-0.07685934 -0.02756647  0.9966608   0.6702519 ]
 [ 0.7472113   0.66023993  0.07588402  3.35129 …[+57 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_static_candidates.py').read().split(\"for phase,ids,cand\")[0];ns={};exec(s,ns);import numpy as np
ids=list(range(0,125));
for p in ['/work/scratch/init_sc1.npz','/work/scratch/init_all.npz']:
 C=np.load(p)['T'][0];b=ns['evaluate'](ids,np.repeat(C[None],len(ids),0),p)
PY"
```
<details><summary>output</summary>

```
/work/scratch/init_sc1.npz mean 0.8859227026928129 median 0.885016998542982 min 0.8764281419122069 max 0.8988192438135912
/work/scratch/init_all.npz mean 0.8860844008737133 median 0.8852598348712968 min 0.876668671076368 max 0.8990140048985744

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);Ci=np.load('/work/scratch/init_all.npz')['T'][0];Cf=np.load('/work/scratch/final_all.npz')['T'][0]
# smooth manip
out=T.copy();out[:125]=Ci;out[592:]=Cf
seg=T[125:592].copy();q=Rot.from_matrix(seg[:,:3,:3]).as_quat()
for i in range(1,len(q)):
 if np.dot(q[i],q[i-1])<0:q[i]*=-1
for j in range(4):q[:,j]=savgol_filter(q[:,j],5,1,mode='interp')
q/=np.linalg.norm(q,axis=1,keepdims=True);R=Rot.from_quat(q).as_matrix();
# blend matrices
blend=np.ones(len(seg));nb=10;blend[:nb]=np.linspace(0,1,nb,endpoint=False);blend[-nb:]=np.linspace(1,0,nb,endpoint=False)[::-1]
Rb=seg[:,:3,:3]*(1-blend[:,None,None])+R*blend[:,None,None];u,s,vt=np.linalg.svd(Rb);R=u@vt
p=seg[:,:3,3].copy();p[:,]=seg[:,:3,3]*(1-blend[:,None])+savgol_filter(seg[:,:3,3],5,1,axis=0,mode='interp')*blend[:,None]
out[125:592,:3,:3]=R;out[125:592,:3,3]=p
# floor min
v=np.array(trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().vertices)[::3];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=pl[:3];d=pl[3]
mn=[]
for i in range(678):
 w=v@out[i,:3,:3].T+out[i,:3,3];mn.append((w@n+d).min())
mn=np.array(mn)
print('floor stats cand',np.percentile(mn,[0,1,5,25,50,75,95,99,100]))
print('worst',[(int(i),round(float(mn[i]),3)) for i in np.argsor …[+220 chars]
```
<details><summary>output</summary>

```
floor stats cand [-0.01432621 -0.01216901 -0.0093411  -0.00527352 -0.00258529  0.003779
  0.00949742  0.01276785  0.01499259]
worst [(444, -0.014), (451, -0.014), (445, -0.013), (443, -0.013), (391, -0.013), (392, -0.012), (390, -0.012), (393, -0.012), (360, -0.011), (389, -0.011), (361, -0.011), (450, -0.011), (394, -0.011), (429, -0.011), (442, -0.01), (449, -0.01), (359, -0.01), (362, -0.01), (326, -0.01), (388, -0.01)]
init floor min -0.0025852901519161087 max sd 0.8824630638710891
final floor min -0.005273521441832685 max sd 0.873193473841752

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_static_candidates.py').read().split(\"for phase,ids,cand\")[0];ns={};exec(s,ns);import numpy as np
C=np.load('/work/scratch/candidate_nosmooth.npz')['T'];ids=list(range(678));a=ns['evaluate'](ids,ns['T'],'base');b=ns['evaluate'](ids,C,'cand');d=b-a
for lo,hi,name in [(0,125,'init'),(125,592,'manip'),(592,678,'final')]:print(name,a[lo:hi].mean(),b[lo:hi].mean(),d[lo:hi].mean(),(d[lo:hi]>0).mean())
print('all',a.mean(),b.mean(),d.mean())
PY"
```
<details><summary>output</summary>

```
base mean 0.8209284069986781 median 0.8287030113368765 min 0.40024162927166035 max 0.9438397910317806
cand mean 0.8350827201708608 median 0.8430152122552006 min 0.4198729262266149 max 0.9365148228488792
init 0.871150751752679 0.8860844008737133 0.0149336491210344 0.952
manip 0.8177918821221195 0.8197202042333351 0.0019283221112156345 0.5438972162740899
final 0.7649628723835931 0.8443744045309528 0.07941153214735941 1.0
all 0.8209284069986781 0.8350827201708608 0.01415431317218276

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/depth_static.py <<'PY'
import numpy as np,torch,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
W,H=1536,1152;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);Twc=np.array(cam['T_world_cam'],dtype=np.float32)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');Kt=torch.tensor(K,device='cuda');Twci=np.linalg.inv(Twc);glctx=dr.RasterizeCudaContext()
def rend(T):
 M=torch.tensor(Twci@T,dtype=torch.float32,device='cuda');v=V@M[:3,:3].T+M[:3,3];zz=v[:,2].clamp_min(1e-5);u=Kt[0,0]*v[:,0]/zz+Kt[0,2];vv=Kt[1,1]*v[:,1]/zz+Kt[1,2];pos=torch.stack([2*u/W-1,2*vv/H-1,zz/10,torch.ones_like(zz)],-1)[None];rast,_=dr.rasterize(glctx,pos,F,resolution=[H,W]);return (rast[0,:,:,3]>0).cpu().numpy(),(rast[0,:,:,2]*10).cpu().numpy()
for ph,ids,cand in [('init',range(0,125),'/work/scratch/init_all.npz'),('final',range(592,678),'/work/scratch/final_all.npz')]:
 C=np.load(cand)['T'][0]; mc,rc=rend(C); diffs=[];dm=[];drz=[]
 with h5py.File('/work/inputs/masks.h5','r') as fm,h5py.File('/pipeline/depth/moge2/aligned.h5','r') as fd:
  for i in ids:
   o=fm['object'][i];p=fm['person'][i];d=cv2.imdecode(fd['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
   m,r=rend(C) # candidate mask static
   sel=o&~p&m
   diffs.append(np.median(d[sel]- …[+251 chars]
```
<details><summary>output</summary>

```
init diff median stats [-0.07912523 -0.05541435  0.00751085  0.05083496  0.07983889] mean 7.297515869140802e-05 moGe med mean 2.9067120000000006 rendered 2.8773146
final diff median stats [-0.06827768 -0.06361619 -0.05775745 -0.05299464 -0.04476577] mean -0.05738089029179064 moGe med mean 2.658343023255814 rendered 2.6675682

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_static_candidates.py').read().split(\"for phase,ids,cand\")[0];ns={};exec(s,ns);import numpy as np,json
C=np.load('/work/scratch/final_all.npz')['T'][0];Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);fwd=Twc[:3,2];ids=list(range(592,678))
for dz in [-.12,-.08,-.06,-.04,-.02,0,.02,.04,.06,.08]:
 X=C.copy();X[:3,3]+=fwd*dz # dz positive moves along camera forward (farther)
 ns['evaluate'](ids,np.repeat(X[None],len(ids),0),f'dz {dz:+.2f}')
PY"
```
<details><summary>output</summary>

```
dz -0.12 mean 0.40492039166708443 median 0.40493957204223435 min 0.40359470821923094 max 0.40645484691910233
dz -0.08 mean 0.535609340853319 median 0.5349012070077539 min 0.5328795273567942 max 0.5441999911820466
dz -0.06 mean 0.6189129696289257 median 0.6176631633981399 min 0.6156959032419406 max 0.632085661954633
dz -0.04 mean 0.7124952131066931 median 0.7106355900063087 min 0.708660653670837 max 0.7310088424437299
dz -0.02 mean 0.80823540127542 median 0.8053821152500834 min 0.8020351526364478 max 0.8355818328406343
dz +0.00 mean 0.8443744045309528 median 0.841554439788512 min 0.8381063087815703 max 0.8703345313795964
dz +0.02 mean 0.7701627999364494 median 0.7683840471906189 min 0.7650645624103299 max 0.7860939825643206
dz +0.04 mean 0.6747484067312177 median 0.6737269657411588 min 0.67 …[+245 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
np.savez('/work/scratch/base600.npz',T=T[600:601])
np.savez('/work/scratch/base0.npz',T=T[0:1])
PY
python /work/scratch/refine_static_all.py --phase final --src /work/scratch/base600.npz --steps 100 --lr 1e-4"
```
<details><summary>output</summary>

```
it 0 loss 0.12596617368134586 dt [-9.999999e-05  9.999998e-05  1.000000e-04] dwdeg [-0.00572957 -0.00572955  0.00572958]
it 10 loss 0.11699120835824446 dt [-0.00111068  0.00098744  0.00110216] dwdeg [ 0.01875514 -0.06037397  0.0630632 ]
it 20 loss 0.11138171838088469 dt [-0.0020535   0.00172898  0.00203943] dwdeg [ 0.00674177 -0.09981615  0.11870123]
it 30 loss 0.10532721335237677 dt [-0.0029883   0.00258575  0.00296686] dwdeg [-0.03175743 -0.13014041  0.17251745]
it 40 loss 0.10045439411293376 dt [-0.0038491   0.00353467  0.00381206] dwdeg [-0.07755081 -0.11006062  0.22118863]
it 50 loss 0.0964598601514643 dt [-0.00455011  0.00429377  0.00446595] dwdeg [-0.14253637 -0.063717    0.2614225 ]
it 60 loss 0.09323223002932289 dt [-0.00532558  0.00460385  0.00507022] dwdeg [-0.21221305 -0.020023 …[+711 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'ids=\\|T0=\\|steps=\\|lr=' /work/scratch/opt_static_init_clean.py /work/scratch/opt_static_final.py /work/scratch/refine_static_clean.py /work/scratch/refine_static.py /work/scratch/refine_static_all.py"
```
<details><summary>output</summary>

```
/work/scratch/opt_static_init_clean.py:40:def optimize(ids,steps=200,lr=0.0003):
/work/scratch/opt_static_init_clean.py:41: B=len(ids); T0=torch.tensor(Tbase[ids[0]:ids[0]+1],device='cuda').repeat(B,1,1); gt=get_masks(ids)
/work/scratch/opt_static_init_clean.py:43: opt=torch.optim.Adam([dt,dw],lr=lr)
/work/scratch/opt_static_init_clean.py:55: ids=list(range(0,124,6)); print('ids',ids)
/work/scratch/opt_static_init_clean.py:58: np.savez('/work/scratch/static_init_clean.npz',T=T,ids=ids,dt=dt,dw=dw,ious=ious)
/work/scratch/opt_static_final.py:40:def optimize(ids,steps=200,lr=0.0003):
/work/scratch/opt_static_final.py:41: B=len(ids); T0=torch.tensor(Tbase[ids[0]:ids[0]+1],device='cuda').repeat(B,1,1); gt=get_masks(ids)
/work/scratch/opt_static_final.py:43: opt=torch.optim.Adam([dt,dw],lr=lr)
 …[+2286 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/outputs && cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF world trajectory for the bar stool.

The released per-frame reconstruction is a useful, but noisy, observation.  This
solver keeps it during manipulation and replaces the two genuinely static phases
(and mildly smooths the manipulated phase) using pooled silhouette evidence.

Usage:
    python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

All phase boundaries below were verified from the video and masks:
  * 0..124     stool is on the floor, untouched/held but not moving
  * 125..591   person drags the stool; heavy hand/body occlusion near 420..560
  * 592..677   stool has been released and is static
The boundaries are deliberately fixed constants so this script is offline and
does not rely on any result produced outside the solver.
"""
import argparse
import json
import os
import random
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


INIT_END = 125       # first moving frame is 125 (verified in frames/masks)
FINAL_START = 592    # stool is stationary from here to the end
BLEND = 10
SEED = 0

# Coarse-to-fine silhouette fitting. Each stage uses a single shared pose.
INIT_STAGES = (
    # (frame stride/selection, resolution divisor, iterations …[+10545 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cd /work && /usr/bin/time -v python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | tee /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cd /work && { time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz ; } 2>&1 | tee /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
Loaded 678 frames, 51247 vertices, 95759 faces
Traceback (most recent call last):
  File "/work/outputs/solve.py", line 290, in <module>
    main()
  File "/work/outputs/solve.py", line 265, in main
    init_pose = fit_shared_pose(
                ^^^^^^^^^^^^^^^^
  File "/work/outputs/solve.py", line 148, in fit_shared_pose
    gt_np = load_visible_masks(args.masks, stage_ids.tolist(), scale)
                               ^^^^^^^^^^
AttributeError: 'Namespace' object has no attribute 'masks'

real	0m3.278s
user	0m7.609s
sys	0m4.349s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/load_visible_masks(args.masks, stage_ids.tolist(), scale)/load_visible_masks(os.path.join(args.inputs, 'masks.h5'), stage_ids.tolist(), scale)/\" /work/outputs/solve.py
python -m py_compile /work/outputs/solve.py
cd /work && { time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz ; } 2>&1 | tee /work/scratch/solve_run.log"
```
<details><summary>output</summary>

```
Loaded 678 frames, 51247 vertices, 95759 faces
Traceback (most recent call last):
  File "/work/outputs/solve.py", line 290, in <module>
    main()
  File "/work/outputs/solve.py", line 265, in main
    init_pose = fit_shared_pose(
                ^^^^^^^^^^^^^^^^
  File "/work/outputs/solve.py", line 179, in fit_shared_pose
    inter = (soft * gt).sum(dim=(1, 2))
             ~~~~~^~~~
RuntimeError: The size of tensor a (21) must match the size of tensor b (8) at non-singleton dimension 0

real	0m8.053s
user	0m12.515s
sys	0m5.165s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace('def render_soft(current_pose):\\n            B = len(stage_ids)\\n            T = current_pose[None].repeat(B, 1, 1)','def render_soft(current_pose, B):\\n            T = current_pose[None].repeat(B, 1, 1)');s=s.replace('soft = render_soft(apply_delta(base_t, center, dt, dw))','soft = render_soft(apply_delta(base_t, center, dt, dw),\\n                                   len(gt_np[start:start + batch_size]))');open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
cd /work && { time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz ; } 2>&1 | tee /work/scratch/solve_run.log"
```
<details><summary>output</summary>

```
Loaded 678 frames, 51247 vertices, 95759 faces
[initial-static] scale=4 it=1/200 dice=0.91494 dt=[-0.0003000000142492354, -0.0003000000142492354, -0.0003000000142492354] dw_deg=[-0.017000000923871994, 0.017000000923871994, 0.017000000923871994]
[initial-static] scale=4 it=40/200 dice=0.93511 dt=[-0.0021100000012665987, 0.00343000004068017, -0.0070500001311302185] dw_deg=[-0.15000000596046448, -0.24699999392032623, -0.028999999165534973]
[initial-static] scale=4 it=80/200 dice=0.93755 dt=[-0.0014299999456852674, 0.0023900000378489494, -0.006930000148713589] dw_deg=[-0.16300000250339508, -0.05700000002980232, -0.17100000381469727]
[initial-static] scale=4 it=120/200 dice=0.93954 dt=[-0.0008399999933317304, 0.001610000035725534, -0.007230000104755163] dw_deg=[-0.16699999570846558, -0.06899999 …[+1182 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace('        del V, F, ones, Kt, Twci, glctx, gt_np, base_t, dt, dw, opt\\n        torch.cuda.empty_cache()','        del Kt, Twci, glctx, gt_np, base_t, dt, dw, opt\\n        torch.cuda.empty_cache()');open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
cd /work && { time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz ; } 2>&1 | tee /work/scratch/solve_run.log"
```
<details><summary>output</summary>

```
Loaded 678 frames, 51247 vertices, 95759 faces
[initial-static] scale=4 it=1/200 dice=0.91494 dt=[-0.0003000000142492354, -0.0003000000142492354, -0.0003000000142492354] dw_deg=[-0.017000000923871994, 0.017000000923871994, 0.017000000923871994]
[initial-static] scale=4 it=40/200 dice=0.93511 dt=[-0.0021100000012665987, 0.00343000004068017, -0.0070500001311302185] dw_deg=[-0.15000000596046448, -0.24699999392032623, -0.028999999165534973]
[initial-static] scale=4 it=80/200 dice=0.93755 dt=[-0.0014299999456852674, 0.0023900000378489494, -0.006930000148713589] dw_deg=[-0.16300000250339508, -0.05700000002980232, -0.17100000381469727]
[initial-static] scale=4 it=120/200 dice=0.93954 dt=[-0.0008399999933317304, 0.001610000035725534, -0.007230000104755163] dw_deg=[-0.16699999570846558, -0.06899999 …[+4499 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/evaluate_final.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, cv2
from scipy.spatial.transform import Rotation as Rot
import nvdiffrast.torch as dr
W,H=1536,1152
np.set_printoptions(precision=4,suppress=True)
inp='/work/inputs'; out='/work/outputs/object_poses_world.npz'
cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],dtype=np.float32);Twc=np.array(cam['T_world_cam'],dtype=np.float32)
z=np.load(inp+'/reconstruction.npz');Tb=z['T_world_mesh'].astype(np.float32);Tr=np.load(out)['T_world_mesh'].astype(np.float32)
mesh=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();verts=np.array(mesh.vertices,dtype=np.float32);faces=np.array(mesh.faces,dtype=np.int32)
V=torch.tensor(verts,device='cuda');F=torch.tensor(faces,device='cuda');Kt=torch.tensor(K,device='cuda');Twci=torch.tensor(np.linalg.inv(Twc),device='cuda');glctx=dr.RasterizeCudaContext()
def render(Tin,depth=False):
 M=torch.tensor(Twci[None].cpu().numpy()@Tin,dtype=torch.float32,device='cuda');B=len(Tin);v=torch.einsum('bij,bvj->bvi',M[:,:3,:3],V[None])+M[:,None,:3,3];zz=v[...,2].clamp_min(1e-5);u=Kt[0,0]*v[...,0]/zz+Kt[0,2];vv=Kt[1,1]*v[...,1]/zz+Kt[1,2];pos=torch.stack([2*u/W-1,2*vv/H-1,zz/10,torch.ones_like(zz)],-1);rast,_=dr.rasterize(glctx,pos,F,resolution=[H,W]);mask=(rast[...,3]>0).cpu().numpy()
 if depth:return mask,(rast[...,2]*10).cpu().numpy()
 return mask
def eval_masks(name,T):
 vals=[];prec=[];rec=[]
 with h5py.File(inp+'/m …[+3444 chars]
```
<details><summary>output</summary>

```
baseline mask all 0.8209284069986781 init 0.871150751752679 manip 0.8177918821221195 final 0.7649628723835931 P 0.9212871664248291 R 0.8802968916136409
result mask all 0.834356696928734 init 0.8823851234208938 manip 0.8197252348037999 final 0.8440001794964578 P 0.9311858629124417 R 0.8869206667255128
baseline floor min mean 0.0002277938463934454 MAE 0.005641761498404639 p95abs 0.011926307511243983 range -0.031194670534711033 0.018192362205104118
result floor min mean -0.0013854387685458807 MAE 0.005598142338854628 p95abs 0.010638723703055452 range -0.014326183293134509 0.01499267038314045
static jitter baseline 0.009981052 0.004009574 result 1.9538327e-06 9.312649e-07
baseline motion manip dp mean/med/p95 0.01204716 0.010721664 0.025024057459086178 rot deg mean/med/p95/max 1.09391718096497 …[+1441 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n '\\bdr\\s*=' /work/scratch/evaluate_final.py"
```
<details><summary>output</summary>

```
44: p=T[:,:3,3];dp=np.linalg.norm(np.diff(p,axis=0),axis=1);r=Rot.from_matrix(T[:,:3,:3]);dr=(r[:-1].inv()*r[1:]).magnitude();return dp,dr
46: dp,dr=motion(T);m=slice(124,592);print(nm,'motion manip dp mean/med/p95',dp[m].mean(),np.median(dp[m]),np.percentile(dp[m],95),'rot deg mean/med/p95/max',np.degrees(dr[m].mean()),np.degrees(np.median(dr[m])),np.degrees(np.percentile(dr[m],95)),np.degrees(dr[m].max()))

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/evaluate_final.py';s=open(p).read();s=s.replace('r=Rot.from_matrix(T[:,:3,:3]);dr=(r[:-1].inv()*r[1:]).magnitude();return dp,dr','r=Rot.from_matrix(T[:,:3,:3]);drotd=(r[:-1].inv()*r[1:]).magnitude();return dp,drotd').replace('dp,dr=motion(T);m=slice(124,592);print(nm,\\'motion manip dp mean/med/p95\\',dp[m].mean(),np.median(dp[m]),np.percentile(dp[m],95),\\'rot deg mean/med/p95/max\\',np.degrees(dr[m].mean()),np.degrees(np.median(dr[m])),np.degrees(np.percentile(dr[m],95)),np.degrees(dr[m].max()))','dp,drotd=motion(T);m=slice(124,592);print(nm,\\'motion manip dp mean/med/p95\\',dp[m].mean(),np.median(dp[m]),np.percentile(dp[m],95),\\'rot deg mean/med/p95/max\\',np.degrees(drotd[m].mean()),np.degrees(np.median(drotd[m])),np.degrees(np.percentile(drotd[m],95)),np.degrees(drotd[m].max()))');open(p,'w').write(s)
PY
python /work/scratch/evaluate_final.py 2>&1 | tee /work/scratch/eval_final.log"
```
<details><summary>output</summary>

```
baseline mask all 0.8209284069986781 init 0.871150751752679 manip 0.8177918821221195 final 0.7649628723835931 P 0.9212871664248291 R 0.8802968916136409
result mask all 0.834356696928734 init 0.8823851234208938 manip 0.8197252348037999 final 0.8440001794964578 P 0.9311858629124417 R 0.8869206667255128
baseline floor min mean 0.0002277938463934454 MAE 0.005641761498404639 p95abs 0.011926307511243983 range -0.031194670534711033 0.018192362205104118
result floor min mean -0.0013854387685458807 MAE 0.005598142338854628 p95abs 0.010638723703055452 range -0.014326183293134509 0.01499267038314045
static jitter baseline 0.009981052 0.004009574 result 1.9538327e-06 9.312649e-07
baseline motion manip dp mean/med/p95 0.01204716 0.010721664 0.025024057459086178 rot deg mean/med/p95/max 1.09391718096497 …[+1061 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# re-use eval script definitions without bottom execution
s=open('/work/scratch/evaluate_final.py').read().split('# save metrics')[0]
# This still executes eval_masks etc, expensive. Instead only exec imports/render definitions before eval_masks call.
head=open('/work/scratch/evaluate_final.py').read().split('def eval_masks')[0]; ns={};exec(head,ns)
import numpy as np,cv2,h5py
render=ns['render'];Tr=ns['Tr'];inp=ns['inp']
flips={'identity':np.eye(4,dtype=np.float32),'z180':np.diag([-1,-1,1,1]).astype(np.float32),'x180':np.diag([1,-1,-1,1]).astype(np.float32),'y180':np.diag([-1,1,-1,1]).astype(np.float32)};ids=[0,60,120,600,650,677]
with h5py.File(inp+'/masks.h5','r') as f:
 for nm,Fl in flips.items():
  vals=[]
  for i in ids:
   m=render((Tr[i]@Fl)[None])[0];o=f['object'][i];p=f['person'][i];pp=cv2.dilate(p.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);g=o&~pp;mm=m&~pp;vals.append(float((mm&g).sum()/max(1,(mm|g).sum())))
  print(nm,round(np.mean(vals),4),[round(x,3) for x in vals])
PY"
```
<details><summary>output</summary>

```
identity 0.8617 [0.884, 0.88, 0.885, 0.841, 0.841, 0.84]
z180 0.2988 [0.369, 0.344, 0.348, 0.244, 0.244, 0.244]
x180 0.351 [0.348, 0.337, 0.348, 0.358, 0.358, 0.358]
y180 0.2139 [0.226, 0.203, 0.212, 0.215, 0.215, 0.214]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/vis_final.py <<'PY'
import numpy as np,torch,trimesh,json,cv2
import nvdiffrast.torch as dr
W,H=1536,1152;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);Twc=np.array(cam['T_world_cam'],dtype=np.float32)
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(np.float32);mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');Kt=torch.tensor(K,device='cuda');Twci=torch.linalg.inv(torch.tensor(Twc,device='cuda'));glctx=dr.RasterizeCudaContext()
def mask(i):
 M=Twci@torch.tensor(T[i],device='cuda');v=V@M[:3,:3].T+M[:3,3];z=v[:,2].clamp_min(1e-5);u=Kt[0,0]*v[:,0]/z+Kt[0,2];vv=Kt[1,1]*v[:,1]/z+Kt[1,2];pos=torch.stack([2*u/W-1,2*vv/H-1,z/10,torch.ones_like(z)],-1)[None];rast,_=dr.rasterize(glctx,pos,F,resolution=[H,W]);return (rast[0,:,:,3]>0).cpu().numpy().astype(np.uint8)
cap=cv2.VideoCapture('/work/inputs/video.mp4');out=[]
for i in [0,90,150,240,330,420,450,510,540,600,650,677]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();m=mask(i);c,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,c,-1,(255,0,0),3);cv2.putText(im,str(i),(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),4);out.append(cv2.resize(im[300:1100,250:1150],(360,320)))
rows=[np.concatenate(out[i:i+3],1) for i in range(0,12,3)];cv2 …[+108 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
p='/work/outputs/object_poses_world.npz';z=np.load(p);T=z['T_world_mesh'];print(z.files,T.shape,T.dtype,np.isfinite(T).all(),np.max(np.abs(T[:,3]-np.array([0,0,0,1]))))
# rigidity
R=T[:,:3,:3];print('orth err',np.max(np.abs(R@R.transpose(0,2,1)-np.eye(3))),'det range',np.linalg.det(R).min(),np.linalg.det(R).max())
# static spread via translations only
for a,b in [(0,125),(592,678)]:print(a,b,'p std',T[a:b,:3,3].std(0),'angle spread deg',np.degrees(np.arccos(np.clip((np.trace(T[a:b,:3,:3],axis1=1,axis2=2)-1)/2,-1,1))).max())
PY"
```
<details><summary>output</summary>

```
['T_world_mesh'] (678, 4, 4) float64 True 0.0
orth err 1.0029958533830374e-05 det range 0.9999957163322168 1.0000002042019898
0 125 p std [0. 0. 0.] angle spread deg 143.8470618050622
592 678 p std [0. 0. 0.] angle spread deg 124.52741509941615

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy,scipy,h5py,cv2,trimesh,torch,nvdiffrast,imageio,PIL
for m in [numpy,scipy,h5py,cv2,trimesh,torch,imageio,PIL]:print(m.__name__,getattr(m,'__version__','?'))
print('nvdiffrast',getattr(nvdiffrast,'__version__','0.4.0'))
PY"
```
<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
h5py 3.16.0
cv2 4.11.0
trimesh 5.1.1
torch 2.5.1+cu124
imageio 2.38.0
PIL 10.2.0
nvdiffrast 0.4.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ffmpeg -version | head -1; python - <<'PY'
import pyrender,open3d
print('pyrender',pyrender.__version__)
print('open3d',open3d.__version__)
PY"
```
<details><summary>output</summary>

```
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
pyrender 0.1.45
open3d 0.20.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];rr=R.from_matrix(T[:,:3,:3]);
for i in [123,124,125,126,133,134,135,581,582,590,591,592,593]:
 j=i+1 if i<677 else i
 print(i,'next dp',np.linalg.norm(T[j,:3,3]-T[i,:3,3]) if j"'!=i else 0,'"'drdeg',np.degrees((rr[i].inv()*rr[j]).magnitude()) if j"'!=i else 0)
PY'
```
<details><summary>output</summary>

```
123 next dp 0.0 drdeg 0.0
124 next dp 0.01137014534503033 drdeg 1.7288885649187165
125 next dp 0.004699816646737919 drdeg 0.45274483048102804
126 next dp 0.005959507638278058 drdeg 0.6040830792286783
133 next dp 0.010005601917761358 drdeg 0.5633046679299972
134 next dp 0.009645433225837025 drdeg 0.49564961550398234
135 next dp 0.009567919090001657 drdeg 0.47774124478674185
581 next dp 0.0011094859221229588 drdeg 0.03946669302770023
582 next dp 0.0018186956292829631 drdeg 0.06052831193587124
590 next dp 0.005857385251148088 drdeg 0.10958937164229599
591 next dp 0.008988278744962525 drdeg 0.8041407079040901
592 next dp 0.0 drdeg 0.0
593 next dp 0.0 drdeg 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Tall bar stool trajectory recovery

## Summary

The released reconstruction is already a strong per-frame tracker: after excluding
the person, its rendered silhouette overlaps the SAM object mask at a full-frame
mean IoU of **0.8209**. Its main weakness in this episode is not a global symmetry
flip. It is frame-to-frame instability, especially in the two phases where the same
physical pose should explain many frames. The final static phase is the clearest
failure: the baseline pose keeps drifting by centimetres even after the person has
released the stool.

I replaced both static phases with one pooled 6-DoF pose per phase and made only a
small, conservative temporal adjustment during manipulation. The resulting
trajectory has mean mask IoU **0.8344**, improves every static frame, keeps floor
contact, and removes the large non-physical jitter in the released pose.

Final output: "'`object_poses_world.npz`, key `T_world_mesh`, shape `(678, 4, 4)`,
dtype `float64`.

## Phases found

| Frames | Phase | Evidence | Confidence |
|---|---|---|---|
| 0-124 | Initial static stool under the desk | Object mask is fixed; baseline translation changes by only a few millimetres; person starts contacting it around frame 96 but does not move it | High |
| 125-419 | Grasped and dragged / turned | Mask and pose begin moving at 125; person and stool are both visible for most of this phase | Medium-high |
| 420-560 | Heavily occluded manipu …[+8597 chars]
```
<details><summary>output</summary>

```
total 116K
-rw-rw-r-- 1 hanzhou hanzhou 8.0K Oct 20 22:23 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.8K Oct 20 22:23 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 22:19 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  86K Oct 20 22:20 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  12K Oct 20 22:19 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cd /work && python /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/object_poses_recheck.npz > /work/scratch/solve_recheck.log 2>&1 && python - <<'PY'
import numpy as np, hashlib
p1='/work/outputs/object_poses_world.npz';p2='/work/scratch/object_poses_recheck.npz'
a=np.load(p1)['T_world_mesh'];b=np.load(p2)['T_world_mesh']
print('shapes',a.shape,b.shape,'dtypes',a.dtype,b.dtype)
print('max_abs_diff',np.max(np.abs(a-b)),'mean_abs_diff',np.mean(np.abs(a-b)))
for p in [p1,p2]:print(p,hashlib.sha256(open(p,'rb').read()).hexdigest())
PY
tail -12 /work/scratch/solve_recheck.log"
```
<details><summary>output</summary>

```
shapes (678, 4, 4) (678, 4, 4) dtypes float64 float64
max_abs_diff 7.7858567237854e-06 mean_abs_diff 6.088838983975505e-07
/work/outputs/object_poses_world.npz 86a17fc31a747b60269dfb9bb23f57b251f582cb9637ceb626c00e8e5608e145
/work/scratch/object_poses_recheck.npz 8736b17bd2bc762d22f3d00d8208465aab968bcf20fa7755cb0403b556ebc6ba
Saved /work/scratch/object_poses_recheck.npz with T_world_mesh (678, 4, 4) float64
Initial static pose:
[[-0.6643884   0.7467423  -0.03104895 -0.05760921]
 [-0.0794215  -0.02923253  0.9964125   0.67166626]
 [ 0.74315566  0.66447073  0.07872915  3.3506246 ]
 [ 0.          0.          0.          1.        ]]
Final static pose:
[[-0.16156213 -0.9868236   0.00885582  0.10395339]
 [ 0.024059    0.00503246  0.99969286  0.73610204]
 [-0.9865696   0.16172382  0.02292867  2. …[+78 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md'
s=open(p).read()
old='The final\\nrun took **63.3 seconds** on the available GPU.'
new='The final\\nrun took **63.3 seconds** on the available GPU (a second verification run on the shared GPU took\\n93.2 seconds; the two outputs agreed to "'`7.8e-6` maximum absolute error).'"'
assert old in s
open(p,'w').write(s.replace(old,new))
PY
ls -lh /work/outputs
python - <<'PY'
import numpy as np
z=np.load('/work/outputs/object_poses_world.npz');T=z['T_world_mesh']
print('keys',z.files,'shape',T.shape,'dtype',T.dtype,'finite',np.isfinite(T).all(),'frames',len(T))
PY"
```
<details><summary>output</summary>

```
keys ['T_world_mesh'] shape (678, 4, 4) dtype float64 finite True frames 678

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import os, glob
for p in glob.glob('/work/outputs/__pycache__/*'):
    os.remove(p)
try:
    os.rmdir('/work/outputs/__pycache__')
except OSError:
    pass
PY
ls -la /work/outputs"
```
<details><summary>output</summary>

```
total 124
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 22:25 .
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 21:58 ..
-rw-rw-r-- 1 hanzhou hanzhou  8298 Oct 20 22:25 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou  1793 Oct 20 22:23 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 87062 Oct 20 22:20 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou 11933 Oct 20 22:19 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');cc=z['T_cam_mesh_coconet'].astype(float);fp=z['T_cam_mesh_foundationpose'].astype(float);T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float)
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);tc=np.linalg.inv(Twc)@T
for name,ref in [('cc',cc),('fp',fp)]:
 ds=[]
 for i in range(678):ds.append(np.degrees((R.from_matrix(ref[i,:3,:3]).inv()*R.from_matrix(tc[i,:3,:3])).magnitude()))
 print(name,'all mean',np.mean(ds),'static init',np.mean(ds[:125]),'manip',np.mean(ds[125:592]),'final',np.mean(ds[592:]))
PY"
```
<details><summary>output</summary>

```
cc all mean 1.15809816307589 static init 1.4609867816889137 manip 1.0140880336601714 final 1.4998615713376648
fp all mean 3.083155039022919 static init 2.2187280162565983 manip 3.422889276505188 final 2.494753747645829

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);
for i in [0,124,125,240,450,591,592,677]:
 a=np.degrees(np.arccos(np.clip(T[i,:3,2]@(-n),-1,1)));print(i,a)
PY"
```
<details><summary>output</summary>

```
0 3.0952833483104762
124 3.0952833483104762
125 4.503354271924611
240 2.354053261561184
450 1.5006105845907138
591 1.3048081149464021
592 1.9672535337608985
677 1.9672535337608985

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'every' /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
15:trajectory has mean mask IoU **0.8344**, improves every static frame, keeps floor
37:1. **Diagnose the baseline.** I rendered the exact textured mesh at every frame with
48:   with 16/15 frames, then full 1536x1152 with every phase frame. Starting from the

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md';s=open(p).read();s=s.replace('trajectory has mean mask IoU **0.8344**, improves every static frame, keeps floor\\ncontact, and removes the large non-physical jitter in the released pose.','trajectory has mean mask IoU **0.8344**, improves both static phases overall (and\\nevery final-static frame), keeps floor contact, and removes the large non-physical\\njitter in the released pose.');open(p,'w').write(s)
PY
sed -n '10,19p' /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
failure: the baseline pose keeps drifting by centimetres even after the person has
released the stool.

I replaced both static phases with one pooled 6-DoF pose per phase and made only a
small, conservative temporal adjustment during manipulation. The resulting
trajectory has mean mask IoU **0.8344**, improves both static phases overall (and
every final-static frame), keeps floor contact, and removes the large non-physical
jitter in the released pose.

Final output: `object_poses_world.npz`, key `T_world_mesh`, shape `(678, 4, 4)`,

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/TOOLS.md';s=open(p).read().replace('Quaternion conversion, Savitzky-Golay smoothing, SVD-adjacent utilities','Quaternion conversion and Savitzky-Golay smoothing');open(p,'w').write(s)
PY"
```
<details><summary>output</summary>

```

```
</details>

**Agent:** Completed the task and produced all required outputs:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, `(678, 4, 4)`, `float64`
- `/work/outputs/solve.py` — offline deterministic solver
- `/work/outputs/REPORT.md` — phases, method, validation table, confidence, runtime
- `/work/outputs/TOOLS.md` — tool versions and dependencies

Validation summary:
- Visible-mask IoU: baseline `0.8209` → result `0.8344`
- Initial static phase: `0.8712` → `0.8824`
- Manipulation: `0.8178` → `0.8197`
- Final static phase: `0.7650` → `0.8440`
- 180° symmetry alternatives scored substantially worse, confirming no global flip
- Final solve runtime: about 63 seconds; independent rerun agreed within `7.8e-6`

The solver was run successfully with the required command and the output was checked for shape, dtype, finiteness, and rigid-transform validity.
