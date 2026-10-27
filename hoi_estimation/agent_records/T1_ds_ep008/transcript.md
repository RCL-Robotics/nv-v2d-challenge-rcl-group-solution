# T1_ds_ep008

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T16:27:42Z

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
/usr/bin/bash -c "wc -l /work/AGENT_TASK.md && sed -n '1,80p' /work/AGENT_TASK.md"
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
and tra …[+6139 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '81,180p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
    tracking, and its refinement by the CARI4D network), mesh-file frame, for `K_pipeline`. Their depth scale is not
    calibrated to the world; `T_world_mesh` already includes a per-frame depth correction.
  - `human_vertices_world` (F,V,3, every 4th vertex), `human_keypoints_world` (F,70,3; MHR keypoints, right hand
    21-41, left hand 42-62) of the reconstructed human in world; `contact_prob_right_left` (F,2).
- `masks.h5` - `object` and `person` (F,1152,1536) boolean SAM2 masks (may be imperfect; you may recompute them).

## Tools you may use
- This container's software (Python, torch + CUDA, nvdiffrast, opencv, trimesh, open3d, ...). The CARI4D pipeline
  modules are installed under `/workspace` (v2d_sam2, v2d_moge, v2d_foundation_pose, v2d_sam3d_body, v2d_cari4d);
  their source an …[+2877 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%f %s bytes\\n' | sort && echo '---task---' && cat /work/inputs/task.txt && echo '---episode---' && cat /work/inputs/episode.json"
```
<details><summary>output</summary>

```
camera.json 824 bytes
episode.json 71 bytes
frame0_reference_human.npz 192027 bytes
ground_plane.json 399 bytes
masks.h5 7680972 bytes
object_mesh.glb 3015924 bytes
reconstruction.npz 35805976 bytes
task.txt 88 bytes
video.mp4 20226570 bytes
---task---
Walk over to the skinny wood chair and push the skinny wood chair into the round table.
---episode---
{
 "episode_index": 8,
 "object": "skinny_wood_chair",
 "frames": 708
}
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json, numpy as np, h5py, cv2, os
for f in ['camera.json','ground_plane.json','episode.json']:
 print('\\n###',f); print(open('/work/inputs/'+f).read())
r=np.load('/work/inputs/reconstruction.npz')
print('\\nkeys',r.files)
for k in r.files:
 a=r[k]; print(k, a.shape, a.dtype)
with h5py.File('/work/inputs/masks.h5','r') as h:
 print('mask keys',list(h.keys()))
 for k in h: print(k,h[k].shape,h[k].dtype, 'sums',h[k][:5].sum(axis=(1,2)))
cap=cv2.VideoCapture('/work/inputs/video.mp4')
print('video',cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
PY"
```
<details><summary>output</summary>

```

### camera.json
{
 "K": [
  [
   820.00048828125,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   820.00048828125,
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
   835.724853515625,
   0.0,
   768.0
  ],
  [
   0.0,
   835.7249145507812,
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
   0.9998110283865015,
   -0.0022516315249189837,
   0.01930900495111284,
   0.08024272576270129
  ],
  [
   0.001766724231537859,
   0.9996835498584349,
   0.025093402079585908,
   0.022644408308588403
  ],
  [
   -0.019359395708952443,
   -0.02505454645197275,
   0.9994986160570059,
   0.03569899161614932
  ],
  [
   0.0,
   0.0,
   0.0,
   1.0
  ]
 ],
 "width": 1536,
 "height": 1152,
 "note": "pinhole, OpenCV camera (x right, y down, z forward); world is the …[+1147 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/analysis /work/outputs /work/tools && python - <<'PY'
import cv2, numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
idxs=list(range(0,708,30))
ims=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read();
 im=cv2.resize(im,(384,288)); cv2.putText(im,str(i),(8,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2); ims.append(im)
while len(ims)%4: ims.append(np.zeros_like(ims[0]))
rows=[np.hstack(ims[j:j+4]) for j in range(0,len(ims),4)]
cv2.imwrite('/work/scratch/analysis/montage_all.jpg',np.vstack(rows))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/traj.py <<'PY'
import numpy as np, h5py
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(float); Tf=r['T_cam_mesh_foundationpose']; Tc=r['T_cam_mesh_coconet']; cp=r['contact_prob_right_left']
# pose chunks
print('n',len(T))
print('idx pos(xyz) rotdelta_prev deg transdelta_prev contact maxobjarea')
prev=None
for i in range(0,len(T),10):
 p=T[i,:3,3]
 if prev is None: rd=td=0
 else:
  rd=np.degrees(R.from_matrix(T[i,:3,:3]@prev[:3,:3].T).magnitude()); td=np.linalg.norm(p-prev[:3,3])
 print(f'{i:4d} {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f} rd {rd:6.2f} td {td:.4f} cp {cp[i,0]:.2f},{cp[i,1]:.2f}')
 prev=T[i]
# norms per-frame
rd=[];td=[]
for i in range(1,len(T)):
 rd.append(np.degrees(R.from_matrix(T[i,:3,:3]@T[i-1,:3,:3].T).magnitude())); td.append(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]))
rd=np.array(rd);td=np.array(td)
print('rotation delta stats',np.percentile(rd,[0,50,90,95,99,100]))
print('trans delta stats',np.percentile(td,[0,50,90,95,99,100]))
print('jump rot>5',np.where(rd>5)[0]+1)
print('jump trans>.02',np.where(td>.02)[0]+1)
print('contact thresholds right >.5',np.where(cp[:,0]>.5)[0][[0,-1]] if (cp[:,0]>.5).any() else None, 'left', np.where(cp[:,1]>.5)[0][[0,-1]] if (cp[:,1]>.5).any() else None)
# dump per-frame summary csv
np.savetxt('/work/scratch/analysis/traj.csv',np.column_stack([np.arange(len(T)),T[:,:3,3],rd if False else np.r_[0,rd],np.r_[0,td],cp]) …[+90 chars]
```
<details><summary>output</summary>

```
n 708
idx pos(xyz) rotdelta_prev deg transdelta_prev contact maxobjarea
   0 +0.1907 +0.7342 +2.6316 rd   0.00 td 0.0000 cp 0.00,0.00
  10 +0.1906 +0.7335 +2.6289 rd   0.00 td 0.0028 cp 0.00,0.00
  20 +0.1903 +0.7318 +2.6226 rd   0.00 td 0.0065 cp 0.00,0.00
  30 +0.1902 +0.7311 +2.6202 rd   0.00 td 0.0025 cp 0.00,0.00
  40 +0.1903 +0.7318 +2.6226 rd   0.00 td 0.0025 cp 0.00,0.00
  50 +0.1905 +0.7331 +2.6273 rd   0.00 td 0.0049 cp 0.00,0.00
  60 +0.1906 +0.7332 +2.6280 rd   0.00 td 0.0007 cp 0.00,0.00
  70 +0.1905 +0.7329 +2.6269 rd   0.00 td 0.0012 cp 0.00,0.00
  80 +0.1921 +0.7305 +2.6227 rd   0.33 td 0.0051 cp 0.00,0.00
  90 +0.1958 +0.7346 +2.6406 rd   0.98 td 0.0187 cp 0.00,0.00
 100 +0.1973 +0.7420 +2.6593 rd   1.09 td 0.0202 cp 0.00,0.00
 110 +0.1975 +0.7432 +2.6637 rd   0.00 td 0.00 …[+4040 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print(type(m),m.units, m.bounds, m.extents)
g=m.to_geometry(); print(type(g),len(g.vertices),len(g.faces),g.extents,g.bounds)
print('visual',type(g.visual),getattr(g.visual,'material',None))
print('geom keys',g.visual.kind)
for i,gg in enumerate(m.geometry.values()): print(i,gg.vertices.shape,gg.faces.shape,gg.extents,gg.visual.kind)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.scene.scene.Scene'> meters [[-0.20215926 -0.23713854 -0.3620992 ]
 [ 0.32071093  0.24068815  0.51493019]] [0.52287018 0.47782668 0.87702939]
<class 'trimesh.base.Trimesh'> 81066 105475 [0.52287018 0.47782668 0.87702939] [[-0.20215926 -0.23713854 -0.3620992 ]
 [ 0.32071093  0.24068815  0.51493019]]
visual <class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x733afa640b50>
geom keys texture
0 (81066, 3) (105475, 3) [0.52287018 0.47782668 0.87702939] texture

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['pyrender','nvdiffrast','pytorch3d','open3d','torch','cv2','scipy','skimage']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__',''))
 except Exception as e: print(m,'NO',e)
PY"
```
<details><summary>output</summary>

```
pyrender OK 0.1.45
nvdiffrast OK 0.4.0
pytorch3d OK 0.7.9
open3d OK 0.20.0
torch OK 2.5.1+cu124
cv2 OK 4.11.0
scipy OK 1.16.3
skimage OK 0.26.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_test.py <<'PY'
import os,sys,json
os.environ['PYOPENGL_PLATFORM']='egl'
import numpy as np, trimesh, pyrender, cv2
from pyrender.constants import RenderFlags
W,H=1536,1152
K=np.array(json.load(open('/work/inputs/camera.json'))['K'])
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
cap=cv2.VideoCapture('/work/inputs/video.mp4')
ren=pyrender.OffscreenRenderer(W,H)
flip=np.diag([1.,-1.,-1.,1.])
for i in [0,100,300,360,400,450,500,550,600,650,700]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read()
 sm=trimesh.Scene(); sm.add_geometry(m,geom_name='obj')
 mesh=pyrender.Mesh.from_trimesh(m,smooth=False)
 scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[2,2,2])
 scene.add(mesh,pose=T[i])
 cam=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=10)
 pose=flip@np.linalg.inv(Twc)
 scene.add(cam,pose=pose)
 color,depth=ren.render(scene,flags=RenderFlags.RGBA)
 alpha=color[:,:,3]>0
 # blend red; dim
 out=img.copy().astype(float)
 out[alpha]=out[alpha]*.35+np.array([255,0,0])*.65
 # draw mask contour baseline
 ctr,_=cv2.findContours(alpha.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(out,ctr,-1,(0,255,255),2)
 cv2.putText(out,f'{i} area {alpha.sum()}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),2) …[+247 chars]
```
<details><summary>output</summary>

```
0 16758 718 703
100 16429 721 703
300 15825 716 705
360 16101 717 706
400 15744 716 705
450 15166 683 698
500 14613 642 693
550 16244 617 693
600 16504 600 692
650 16319 601 691
700 16291 602 692

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,glob,os
fs=sorted(glob.glob('/work/scratch/analysis/render_*.jpg')); ims=[]
for f in fs:
 im=cv2.imread(f); ims.append(cv2.resize(im,(512,384)))
rows=[np.hstack(ims[i:i+3]) for i in range(0,len(ims),3)]
cv2.imwrite('/work/scratch/analysis/render_montage.jpg',np.vstack(rows))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/shape_base.py", line 289, in vstack
    return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1536 and the array at index 3 has size 1024

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,glob
fs=sorted(glob.glob('/work/scratch/analysis/render_*.jpg')); ims=[cv2.resize(cv2.imread(f),(512,384)) for f in fs]
while len(ims)%3: ims.append(np.zeros_like(ims[0]))
cv2.imwrite('/work/scratch/analysis/render_montage.jpg',np.vstack([np.hstack(ims[i:i+3]) for i in range(0,len(ims),3)]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh,cv2
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float); c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=m.vertices
for i in [0,100,300,400,500,600,700]:
 p=(np.linalg.inv(Twc)@T[i]@np.c_[v,np.ones(len(v))].T).T[:,:3]
 uv=(K@p.T).T; uv=uv[:,:2]/uv[:,2:3]
 print(i, uv.min(0),uv.max(0),uv.mean(0), 'depth',p[:,2].min(),p[:,2].max())
PY"
```
<details><summary>output</summary>

```
0 [670.00749195 661.12175439] [861.52606945 960.44891865] [778.48868201 763.24269294] depth 2.3467741477875848 3.01134475236467
100 [673.2742952  661.31189205] [863.3681161  958.39722593] [780.13267463 763.28566225] depth 2.3795245964863327 3.031310030771034
300 [668.7499048  663.49253058] [855.93360888 955.84790023] [776.51596481 763.33154162] depth 2.4053249969490866 3.0681305358736846
400 [669.37758656 663.19932898] [855.69243839 953.92023231] [775.94058223 762.53738953] depth 2.4180026173128537 3.083078110800643
500 [600.46947707 651.32319771] [773.31903422 917.10915458] [693.90044508 742.55973554] depth 2.625759938331713 3.2971738981510788
600 [559.66780389 649.88255898] [706.86194064 903.51547203] [630.06026364 738.34711831] depth 2.726185462922123 3.3181851600193286
700 [561.1387852 …[+114 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as h:
 m=h['object'][:]
for i in [0,100,300,400,500,600,700]:
 y,x=np.where(m[i]); print(i,'bbox',x.min(),y.min(),x.max(),y.max(),'cent',x.mean(),y.mean(),'area',len(x))
PY"
```
<details><summary>output</summary>

```
0 bbox 664 660 860 966 cent 775.7230575065847 769.1270302897278 area 18224
100 bbox 719 660 861 966 cent 792.0893599892661 761.410707097813 area 14906
300 bbox 663 686 809 966 cent 745.75315084679 792.5478534856243 area 10156
400 bbox 663 668 836 965 cent 764.2429843038663 776.9246449684039 area 14717
500 bbox 593 651 774 924 cent 692.1007371007371 749.8387011089713 area 15059
600 bbox 535 651 707 905 cent 631.2104479390041 737.8574577078866 area 16788
700 bbox 535 651 708 905 cent 632.4481916766025 732.8564128768523 area 17613

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_iou.py <<'PY'
import os,json,numpy as np,h5py,cv2,trimesh,pyrender
os.environ['PYOPENGL_PLATFORM']='egl'
from pyrender.constants import RenderFlags
W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); flip=np.diag([1.,-1.,-1.,1.])
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mesh=pyrender.Mesh.from_trimesh(m,smooth=False)
ren=pyrender.OffscreenRenderer(W,H); scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[2,2,2]); scene.add(mesh)
cam=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=10); scene.add(cam,pose=flip@np.linalg.inv(Twc))
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; h=h5py.File('/work/inputs/masks.h5','r'); masks=h['object']
for i in [0,100,300,400,500,600,700]:
 scene.set_pose(mesh,T[i]); color,d=ren.render(scene,flags=RenderFlags.RGBA); a=color[:,:,3]>0; b=masks[i]
 inter=(a&b).sum(); union=(a|b).sum(); print(i, a.sum(),b.sum(),inter/union,inter/a.sum(),inter/b.sum())
PY
python /work/scratch/analysis/render_iou.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/render_iou.py", line 11, in <module>
    scene.set_pose(mesh,T[i]); color,d=ren.render(scene,flags=RenderFlags.RGBA); a=color[:,:,3]>0; b=masks[i]
    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/pyrender/scene.py", line 477, in set_pose
    raise ValueError('Node must already be in scene')
ValueError: Node must already be in scene

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/scene.add(mesh)/node=scene.add(mesh)/; s/scene.set_pose(mesh,T\\[i\\])/scene.set_pose(node,T[i])/\" /work/scratch/analysis/render_iou.py && python /work/scratch/analysis/render_iou.py"
```
<details><summary>output</summary>

```
0 16758 18224 0.08971403650862875 0.17185821697099893 0.15803336259877085
100 16429 14906 0.09257322175732217 0.16160447988313348 0.1781161948208775
300 15825 10156 0.04580767218129855 0.07191153238546603 0.11205198897203623
400 15744 14717 0.0789911799086111 0.14164126016260162 0.15152544676224775
500 14613 15059 0.1173369483355927 0.21323479093957434 0.20691945016269342
600 16504 16788 0.21238164603058995 0.35336888027144936 0.34739099356683345
700 16291 17613 0.19916528136384537 0.34565097292983854 0.3197070345767331

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/flip@np.linalg.inv(Twc)/Twc@flip/' /work/scratch/analysis/render_iou.py /work/scratch/analysis/render_test.py && python /work/scratch/analysis/render_iou.py"
```
<details><summary>output</summary>

```
0 16369 18224 0.7934985483201991 0.9349990836336979 0.8398266022827041
100 16058 14906 0.6850239442751415 0.7839083322954291 0.8444921508117537
300 15533 10156 0.4884408134886146 0.5427155089165003 0.8300512012603387
400 15453 14717 0.7145942259604455 0.8136931340192843 0.854386084120405
500 14415 15059 0.729897875337481 0.8627124523066251 0.825818447440069
600 16142 16788 0.7980779731353064 0.9054640069384216 0.8706218727662616
700 15913 17613 0.8257365354244949 0.9528687236850374 0.8608982001930392

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/scratch/analysis/render_test.py >/dev/null && python - <<'PY'
import cv2,numpy as np,glob
fs=sorted(glob.glob('/work/scratch/analysis/render_*.jpg')); ims=[cv2.resize(cv2.imread(f),(512,384)) for f in fs]
while len(ims)%3: ims.append(np.zeros_like(ims[0]))
cv2.imwrite('/work/scratch/analysis/render_montage2.jpg',np.vstack([np.hstack(ims[i:i+3]) for i in range(0,len(ims),3)]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
S,b=trimesh.bounds.oriented_bounds(m)
print('S\\n',S); print('bounds',b); print('center mesh',m.bounds.mean(0),'centroid',m.centroid)
print('S center', S@np.r_[m.bounds.mean(0),1])
PY"
```
<details><summary>output</summary>

```
S
 [[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 -1.77480280e-03]
 [ 1.00000000e+00  0.00000000e+00  6.12323400e-17 -5.92758358e-02]
 [ 6.12323400e-17  0.00000000e+00 -1.00000000e+00  7.64154941e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
bounds [0.47782668 0.52287018 0.87702939]
center mesh [0.05927584 0.0017748  0.07641549] centroid [1.28320274e-10 2.10113138e-10 5.66404016e-10]
S center [0.00000000e+00 3.46944695e-17 0.00000000e+00 1.00000000e+00]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); A=r['T_world_mesh']; F=r['T_cam_mesh_foundationpose']; C=r['T_cam_mesh_coconet']
for name,X in [('FP',F),('CoCo',C)]:
 rel=[]
 for i in range(len(A)):
  rel.append(np.degrees(R.from_matrix(A[i,:3,:3].T@X[i,:3,:3]).magnitude()))
 rel=np.array(rel); print(name,'rot rel percentiles',np.percentile(rel,[0,10,50,90,95,99,100]))
 for i in range(0,708,50): print(i,round(rel[i],1),end='; ')
 print()
 print('nan',np.isnan(X).any(axis=(1,2)).sum())
PY"
```
<details><summary>output</summary>

```
FP rot rel percentiles [0.3625829  1.35527388 2.62081147 3.90912177 4.37284117 5.58906801
 7.26859401]
0 1.9; 50 1.9; 100 4.4; 150 6.2; 200 2.1; 250 2.1; 300 3.0; 350 3.0; 400 3.7; 450 3.4; 500 3.0; 550 1.9; 600 0.8; 650 2.2; 700 1.9; 
nan 0
CoCo rot rel percentiles [3.0930099  3.09519269 3.2258936  3.22837935 3.22865179 3.22945977
 3.22982961]
0 3.2; 50 3.2; 100 3.2; 150 3.2; 200 3.2; 250 3.2; 300 3.2; 350 3.2; 400 3.2; 450 3.2; 500 3.2; 550 3.1; 600 3.1; 650 3.1; 700 3.1; 
nan 0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_raw.py <<'PY'
import os,json,numpy as np,cv2,trimesh,pyrender
os.environ['PYOPENGL_PLATFORM']='egl'
W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); flip=np.diag([1.,-1.,-1.,1.])
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mesh=pyrender.Mesh.from_trimesh(m,smooth=False)
ren=pyrender.OffscreenRenderer(W,H); scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[2,2,2]); node=scene.add(mesh)
cam=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=10); scene.add(cam,pose=Twc@flip)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,100,400,500,600,700]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read(); scene.set_pose(node,T[i]); col,dep=ren.render(scene); a=col[:,:,3]>0
 out=img.copy(); out[a]=col[:,:3][a]
 # side by side crop x 480:1000, y 600:1000
 crop=lambda x: cv2.resize(x[600:1000,500:1000],(500,400))
 cv2.imwrite(f'/work/scratch/analysis/raw_{i:03d}.jpg',np.hstack([crop(img),crop(out)]))
PY
python /work/scratch/analysis/render_raw.py && python - <<'PY'
import cv2,glob,numpy as np
ims=[cv2.imread(f) for f in sorted(glob.glob('/work/scratch/analysis/raw_*.jpg'))]
for i in range(0,len(ims),2):
 cv2.imwrite(f'/work/scratch/analysis/raw_pair_{i//2}.jpg',np.hstack(ims[i:i+2]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/render_raw.py", line 11, in <module>
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read(); scene.set_pose(node,T[i]); col,dep=ren.render(scene); a=col[:,:,3]>0
                                                                                                                   ~~~^^^^^^^
IndexError: index 3 is out of bounds for axis 2 with size 3

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/col,dep=ren.render(scene)/col,dep=ren.render(scene,flags=pyrender.constants.RenderFlags.RGBA)/' /work/scratch/analysis/render_raw.py && python /work/scratch/analysis/render_raw.py && python - <<'PY'
import cv2,glob,numpy as np
ims=[cv2.imread(f) for f in sorted(glob.glob('/work/scratch/analysis/raw_*.jpg'))]
for i in range(0,len(ims),2): cv2.imwrite(f'/work/scratch/analysis/raw_pair_{i//2}.jpg',np.hstack(ims[i:i+2]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/render_raw.py", line 12, in <module>
    out=img.copy(); out[a]=col[:,:3][a]
                           ~~~~~~~~~^^^
IndexError: boolean index did not match indexed array along dimension 1; dimension is 3 but corresponding boolean dimension is 1536

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/col\\[:,:3\\]\\[a\\]/col[...,:3][a]/' /work/scratch/analysis/render_raw.py && python /work/scratch/analysis/render_raw.py && python - <<'PY'
import cv2,glob,numpy as np
ims=[cv2.imread(f) for f in sorted(glob.glob('/work/scratch/analysis/raw_*.jpg'))]
for i in range(0,len(ims),2): cv2.imwrite(f'/work/scratch/analysis/raw_pair_{i//2}.jpg',np.hstack(ims[i:i+2]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/analysis/render_raw.py /work/scratch/analysis/render_flat.py && sed -i 's/col,dep=ren.render(scene,flags=pyrender.constants.RenderFlags.RGBA)/col,dep=ren.render(scene,flags=pyrender.constants.RenderFlags.RGBA + pyrender.constants.RenderFlags.FLAT)/' /work/scratch/analysis/render_flat.py && sed -i 's#raw_{i:03d}.jpg#flat_{i:03d}.jpg#' /work/scratch/analysis/render_flat.py && python /work/scratch/analysis/render_flat.py && python - <<'PY'
import cv2,glob,numpy as np
ims=[cv2.imread(f) for f in sorted(glob.glob('/work/scratch/analysis/flat_*.jpg'))]
for i in range(0,len(ims),2): cv2.imwrite(f'/work/scratch/analysis/flat_pair_{i//2}.jpg',np.hstack(ims[i:i+2]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mat=m.visual.material; print(mat)
for k in ['baseColorFactor','metallicFactor','roughnessFactor','doubleSided','alphaMode']: print(k,getattr(mat,k,None))
for k in ['baseColorTexture','emissiveTexture','metallicRoughnessTexture','normalTexture','occlusionTexture']:
 v=getattr(mat,k,None); print(k,type(v),getattr(v,'size',None))
 if v is not None: print(v.getdata().shape, v.getdata().min(),v.getdata().max(),v.getdata().mean(axis=(0,1)))
PY"
```
<details><summary>output</summary>

```
<trimesh.visual.material.PBRMaterial object at 0x7bb4aad66410>
baseColorFactor [255 255 255 255]
metallicFactor 0.0
roughnessFactor 1.0
doubleSided False
alphaMode None
baseColorTexture <class 'PIL.Image.Image'> (256, 256)
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
AttributeError: 'ImagingCore' object has no attribute 'shape'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); t=np.array(m.visual.material.baseColorTexture); print(t.shape,t.min(),t.max(),t.mean((0,1)))
from PIL import Image
Image.fromarray(t).save('/work/scratch/analysis/texture.png')
PY"
```
<details><summary>output</summary>

```
(256, 256, 3) 0 255 [89.44584656 73.79173279 57.19050598]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,pyrender,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); pm=pyrender.Mesh.from_trimesh(m,smooth=False)
print(len(pm.primitives))
for p in pm.primitives: print(p.material, p.material.baseColorTexture is None, p.material.baseColorFactor)
PY"
```
<details><summary>output</summary>

```
1
<pyrender.material.MetallicRoughnessMaterial object at 0x762e9848d750> False [1. 1. 1. 1.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); print(m.visual.uv.shape,m.visual.uv.min(0),m.visual.uv.max(0)); print(m.visual.kind)
PY"
```
<details><summary>output</summary>

```
(81066, 2) [0. 0.] [0.9995656  0.99957007]
texture

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
for i in [0,300,350,370,400,450,500,550,600,650,700]:
 rel=R.from_matrix(T[i,:3,:3]@T[0,:3,:3].T)
 print(i,'rotvec',np.round(rel.as_rotvec()*180/np.pi,1),'angle',round(rel.magnitude()*180/np.pi,1),'euler xyz',np.round(rel.as_euler('xyz',degrees=True),1))
PY"
```
<details><summary>output</summary>

```
0 rotvec [ 0. -0. -0.] angle 0.0 euler xyz [ 0.  0. -0.]
300 rotvec [-0.  -1.9  0.3] angle 1.9 euler xyz [-0.  -1.9  0.3]
350 rotvec [-0.  -1.4  0.1] angle 1.4 euler xyz [-0.  -1.4  0.1]
370 rotvec [ 0.3 -1.1  0.5] angle 1.2 euler xyz [ 0.3 -1.1  0.5]
400 rotvec [ 0.1 -1.3  0.2] angle 1.3 euler xyz [ 0.1 -1.3  0.2]
450 rotvec [-0.5  0.8  0.4] angle 1.0 euler xyz [-0.5  0.8  0.3]
500 rotvec [0.4 4.3 0.2] angle 4.4 euler xyz [0.5 4.3 0.3]
550 rotvec [-1.  18.1  0.1] angle 18.2 euler xyz [-1.  18.1 -0.1]
600 rotvec [-2.6 29.2 -0.1] angle 29.3 euler xyz [-2.9 29.2 -0.9]
650 rotvec [-1.2 28.7  0.6] angle 28.7 euler xyz [-1.2 28.7  0.4]
700 rotvec [-1.2 28.7  0.6] angle 28.7 euler xyz [-1.2 28.7  0.4]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -100 && echo '--- dirs ---' && find /pipeline -maxdepth 2 -type d -print"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1847
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1981
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/05_object_pose.json 1772
/pipeline/.stages/06_coconet.json 2828
/pipeline/.stages/07_refinement.json 2130
/pipeline/.stages/08_render.json 2525
/pipeline/depth/moge2/depth_report.json 1809
/pipeline/depth/moge2/aligned.h5 896357304
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 75455
/pipeline/depth/moge2/raw.npy 2505572480
/pipeline/export/episode_000008/edex 562
/pipeline/export/episode_000008/wild_export.json 2689
/pipeline/foundationpose/episode_000008.pkl 203150
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 5415659 …[+807 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/depth/moge2/depth_report.json && echo '---' && cat /pipeline/.stages/04_depth_alignment.json"
```
<details><summary>output</summary>

```
{
  "backend": "moge2",
  "batchSize": 8,
  "cameraPolicy": "first_frame_export_per_frame_depth",
  "depthM": {
    "max": 9.909563064575195,
    "min": 1.41111159324646
  },
  "deviceCount": 1,
  "devices": [
    "cuda"
  ],
  "elapsedSeconds": 120.74722247710451,
  "firstFrameIntrinsicsPx": [
    [
      835.724853515625,
      0.0,
      768.0
    ],
    [
      0.0,
      835.7249145507812,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "focalXPx": {
    "max": 840.4901733398438,
    "mean": 826.8882446289062,
    "min": 816.7230224609375
  },
  "focalYPx": {
    "max": 840.490234375,
    "mean": 826.8882446289062,
    "min": 816.7230224609375
  },
  "fps": 30.0,
  "frames": 708,
  "framesPerSecond": 5.863488910763537,
  "height": 1152,
  "intrinsicsFile": {
     …[+2860 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for f in ['/pipeline/depth/moge2/aligned.h5','/pipeline/inference/materialized_inputs.h5','/pipeline/work/masks_k0.h5']:
 print('\\n',f)
 with h5py.File(f,'r') as h:
  def show(name,obj):
   if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype, dict(obj.attrs))
  h.visititems(show)
PY"
```
<details><summary>output</summary>

```

 /pipeline/depth/moge2/aligned.h5
aligned/front_stereo_camera_left (708,) object {}
alignment/front_stereo_camera_left/scale (708,) float32 {}
alignment/front_stereo_camera_left/shift (708,) float32 {}
alignment/front_stereo_camera_left/valid_count (708,) int32 {}
frame_names/front_stereo_camera_left (708,) object {}
raw/front_stereo_camera_left (708,) object {}

 /pipeline/inference/materialized_inputs.h5
K_rois (708, 3, 3) float32 {}
bboxes (708, 4) float32 {}
identity_json () object {}
input_rgbs (708, 3, 224, 224) float32 {}
input_xyz (708, 6, 224, 224) float32 {}
observed_human_masks (708, 224, 224) bool {}
observed_object_masks (708, 224, 224) bool {}
render_rgbs (708, 3, 224, 224) float32 {}
render_xyz (708, 6, 224, 224) float32 {}

 /pipeline/work/masks_k0.h5
episode_000008/000000 …[+90603 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
f='/pipeline/depth/moge2/aligned.h5'; h=h5py.File(f,'r')
g='front_stereo_camera_left'
for i in [0,100,400,600]:
 print('\\nframe',i)
 for k in ['alignment/'+g+'/scale','alignment/'+g+'/shift','alignment/'+g+'/valid_count']:
  print(k,h[k][i])
 for k in ['aligned/'+g,'raw/'+g]:
  o=h[k][i]; print(k,type(o),getattr(o,'shape',None),getattr(o,'dtype',None));
  if isinstance(o,np.ndarray): print(o.min(),o.max(),o.mean(),o[:2,:3])
  elif isinstance(o,h5py.Group): print(list(o.keys()))
  else: print(o)
 h.close(); break
PY"
```
<details><summary>output</summary>

```
alignment/front_stereo_camera_left/shift 0.0
alignment/front_stereo_camera_left/valid_count 56334
aligned/front_stereo_camera_left <class 'numpy.ndarray'> (669957,) uint8
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
h=h5py.File('/pipeline/depth/moge2/aligned.h5','r'); g=h['aligned/front_stereo_camera_left']; print(g.shape,g.dtype,dict(g.attrs)); print(dict(h.attrs)); print(dict(h['raw/front_stereo_camera_left'].attrs))
PY"
```
<details><summary>output</summary>

```
(708,) object {}
{'complete': True, 'depth_alignment_input_identity_json': '{"depth_backend":"moge2","depth_model_id":"Ruicheng/moge-2-vitl-normal","depth_model_revision":"b135031bae30b5ac2ae141a0e68717795ce38340","depth_source_commit":"925b8ed835a7a9cdb7578ba15c658a0afc969030","direct_mhr":{"mtime_ns":1791206296942796304,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000008/mhr_init/episode_000008.pkl","size":3160083},"export":{"mtime_ns":1791206097890797064,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000008/export/episode_000008/wild_export.json","size":2689},"raw_depth":{"mtime_ns":1791205965273797570,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000008/depth/moge2/raw.npy","size":2505572480},"raw_depth_report":{"mtime_ns":1791205968233797558,"path":"/home/hanzhou/Projec …[+242 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
h=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
for i in [0,100,400,600]:
 b=h['aligned/front_stereo_camera_left'][i]; im=cv2.imdecode(b,cv2.IMREAD_UNCHANGED); print(i,len(b),im.shape,im.dtype,im.min(),im.max(),im.mean())
PY"
```
<details><summary>output</summary>

```
0 669957 (1152, 1536) uint16 1851 10254 4446.4832599781175
100 672124 (1152, 1536) uint16 1895 10811 4580.7816970260055
400 662302 (1152, 1536) uint16 1814 10494 4310.592500474718
600 653101 (1152, 1536) uint16 1812 10641 4380.173822473596

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
g=json.load(open('/work/inputs/ground_plane.json')); n=np.array(g['plane'][:3]); d=g['plane'][3]
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=np.c_[m.vertices,np.ones(len(m.vertices))]
# bottom y points for a sample pose identify local indices? use transformed each
st=[]
for i in range(len(T)):
 w=(T[i]@v.T).T[:,:3]; sd=w@n+d; st.append([sd.min(),np.percentile(sd,1),np.percentile(sd,5),sd.max(),w[:,1].min(),w[:,1].max()])
st=np.array(st)
for i in [0,100,300,350,400,450,500,550,600,650,700]: print(i,np.round(st[i],4))
print('min plane pct',np.percentile(st[:,0],[0,50,90,100])); print('1pct',np.percentile(st[:,1],[0,50,90,100]))
PY"
```
<details><summary>output</summary>

```
0 [0.0032 0.0451 0.1843 0.8503 0.3956 1.2503]
100 [-0.0034  0.0351  0.1735  0.8443  0.3999  1.2541]
300 [-0.016   0.0269  0.1651  0.8299  0.4126  1.2662]
350 [-0.0194  0.023   0.1617  0.8272  0.4155  1.2699]
400 [-0.0184  0.0248  0.1637  0.8282  0.4136  1.2678]
450 [-0.006   0.0337  0.1717  0.8392  0.4004  1.2531]
500 [-0.012   0.0317  0.1709  0.8356  0.4003  1.2547]
550 [-0.004   0.0364  0.175   0.8489  0.3912  1.2446]
600 [-0.0214  0.0269  0.1637  0.8471  0.3921  1.2456]
650 [-0.0061  0.0357  0.1737  0.8497  0.3898  1.2417]
700 [-0.0068  0.035   0.1729  0.849   0.3904  1.2423]
min plane pct [-0.04194126 -0.0122666   0.00435164  0.01106655]
1pct [0.00247681 0.03131188 0.04624645 0.05424362]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float); cp=r['contact_prob_right_left']
p=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1); a=np.array([np.degrees(R.from_matrix(T[i,:3,:3]@T[i-1,:3,:3].T).magnitude()) for i in range(1,len(T))])
for i in range(330,431):
 if i%5==0: print(i,'xyz',np.round(T[i,:3,3],4),'vel',round(p[i-1],4),'rot',round(a[i-1],2),'cp',np.round(cp[i],2))
print('--- final')
for i in range(580,708):
 if i%5==0: print(i,'xyz',np.round(T[i,:3,3],4),'vel',round(p[i-1],4),'rot',round(a[i-1],2),'cp',np.round(cp[i],2))
PY"
```
<details><summary>output</summary>

```
330 xyz [0.1795 0.7543 2.6879] vel 0.0167 rot 0.35 cp [0. 0.]
335 xyz [0.1825 0.7544 2.6923] vel 0.0146 rot 0.09 cp [0. 0.]
340 xyz [0.1829 0.7517 2.6866] vel 0.0059 rot 0.32 cp [0. 0.]
345 xyz [0.1858 0.7528 2.6875] vel 0.0019 rot 0.22 cp [0.06 0.18]
350 xyz [0.1871 0.7539 2.6854] vel 0.0084 rot 0.4 cp [0.01 0.7 ]
355 xyz [0.1869 0.7488 2.6717] vel 0.0062 rot 0.15 cp [0.04 0.82]
360 xyz [0.1905 0.7497 2.6751] vel 0.0091 rot 0.52 cp [0.41 0.77]
365 xyz [0.1899 0.7472 2.6732] vel 0.0046 rot 0.33 cp [1. 1.]
370 xyz [0.1933 0.7492 2.6779] vel 0.0011 rot 0.1 cp [1. 1.]
375 xyz [0.1878 0.7516 2.707 ] vel 0.0006 rot 0.0 cp [1. 1.]
380 xyz [0.1878 0.7517 2.7074] vel 0.0 rot 0.0 cp [1. 1.]
385 xyz [0.1878 0.7521 2.7088] vel 0.0005 rot 0.0 cp [1. 1.]
390 xyz [0.1878 0.752  2.7083] vel 0.0005 rot 0. …[+2182 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4'); ids=list(range(360,421,5)); ims=[]
for i in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); im=cv2.resize(im,(480,360)); cv2.putText(im,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2); ims.append(im)
while len(ims)%4: ims.append(np.zeros_like(ims[0]))
cv2.imwrite('/work/scratch/analysis/motion_detail.jpg',np.vstack([np.hstack(ims[i:i+4]) for i in range(0,len(ims),4)]))
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
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
for i in range(365,386):
 if i>0:
  rd=np.degrees(R.from_matrix(T[i,:3,:3]@T[i-1,:3,:3].T).magnitude()); td=np.linalg.norm(T[i,:3,3]-T[i-1,:3,3])
 else:rd=td=0
 print(i, np.round(T[i,:3,3],5),round(td,5),round(rd,2))
PY"
```
<details><summary>output</summary>

```
365 [0.18988 0.74721 2.67318] 0.00463 0.33
366 [0.19142 0.74833 2.67326] 0.00191 0.25
367 [0.19223 0.75051 2.67465] 0.00271 0.62
368 [0.19284 0.7492  2.67331] 0.00197 0.12
369 [0.19271 0.74837 2.6782 ] 0.00496 0.25
370 [0.19334 0.74916 2.67787] 0.00106 0.1
371 [0.19148 0.7513  2.67926] 0.00316 0.34
372 [0.19024 0.75116 2.67872] 0.00137 0.13
373 [0.19322 0.74861 2.674  ] 0.00614 0.22
374 [0.18773 0.75146 2.7064 ] 0.03299 0.32
375 [0.18775 0.75161 2.70695] 0.00057 0.0
376 [0.18775 0.75161 2.70695] 0.0 0.0
377 [0.18776 0.75166 2.70715] 0.0002 0.0
378 [0.18777 0.75173 2.70739] 0.00025 0.0
379 [0.18777 0.75173 2.70739] 0.0 0.0
380 [0.18777 0.75174 2.70743] 4e-05 0.0
381 [0.18779 0.7519  2.70801] 0.00061 0.0
382 [0.1878  0.75194 2.70817] 0.00016 0.0
383 [0.18782 0.75211 2.70879] 0.00064 0.0
384  …[+80 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4'); ids=[350,360,365,370,373,374,375,376,380,385,390,395,400,405,410]; ims=[]
for i in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); im=im[580:1020,570:930]; im=cv2.resize(im,(360,440)); cv2.putText(im,str(i),(8,32),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,255,255),2); ims.append(im)
while len(ims)%5: ims.append(np.zeros_like(ims[0]))
cv2.imwrite('/work/scratch/analysis/static_transition.jpg',np.vstack([np.hstack(ims[i:i+5]) for i in range(0,len(ims),5)]))
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
h=h5py.File('/work/inputs/masks.h5','r'); o=h['object']; p=h['person']
for i in range(340,421):
 y,x=np.where(o[i]); py,px=np.where(p[i]);
 # centroid visible not overlap person
 vis=o[i]&~p[i]; yv,xv=np.where(vis)
 if i%2==0: print(i,'bbox',x.min(),y.min(),x.max(),y.max(),'cent',round(x.mean(),2),round(y.mean(),2),'area',len(x),'vis',len(xv),'vcent',round(xv.mean(),2) if len(xv) else -1,round(yv.mean(),2) if len(yv) else -1)
PY"
```
<details><summary>output</summary>

```
340 bbox 663 673 831 966 cent 757.63 782.74 area 12746 vis 12710 vcent 757.48 782.96
342 bbox 663 671 833 965 cent 759.38 782.26 area 13358 vis 13354 vcent 759.36 782.29
344 bbox 663 669 834 966 cent 759.04 781.93 area 13328 vis 13196 vcent 758.53 782.44
346 bbox 663 670 837 966 cent 759.76 781.89 area 13526 vis 13358 vcent 759.09 782.39
348 bbox 663 670 838 966 cent 760.34 781.36 area 13714 vis 13524 vcent 759.59 782.12
350 bbox 663 672 838 966 cent 760.36 781.35 area 13771 vis 13760 vcent 760.32 781.4
352 bbox 663 672 839 966 cent 760.54 781.08 area 13799 vis 13791 vcent 760.51 781.13
354 bbox 663 672 839 966 cent 760.66 781.17 area 13849 vis 13837 vcent 760.61 781.24
356 bbox 663 671 840 966 cent 761.5 781.55 area 13952 vis 13914 vcent 761.41 781.7
358 bbox 663 672 839 966 cent 760.84 7 …[+2668 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/eval_static.py <<'PY'
import os,json,numpy as np,h5py,cv2,trimesh,pyrender
os.environ['PYOPENGL_PLATFORM']='egl'
from scipy.spatial.transform import Rotation as R
from pyrender.constants import RenderFlags
W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); flip=np.diag([1.,-1.,-1.,1.])
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mesh=pyrender.Mesh.from_trimesh(m,smooth=False)
ren=pyrender.OffscreenRenderer(W,H); scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[2,2,2]); node=scene.add(mesh)
cam=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=10); scene.add(cam,pose=Twc@flip)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
h=h5py.File('/work/inputs/masks.h5','r')
def meanpose(idx):
 t=np.median(T[idx,:3,3],axis=0)
 rot=R.from_matrix(T[idx,:3,:3]).mean().as_matrix()
 out=np.eye(4);out[:3,:3]=rot;out[:3,3]=t;return out
poses={'A_pre':meanpose(np.arange(0,374)),'B_hold':meanpose(np.arange(378,400)),'all':meanpose(np.arange(0,400)),'base0':T[0],'base100':T[100],'base350':T[350]}
for k,v in poses.items(): print(k,'pos',v[:3,3],'rotrel',np.degrees(R.from_matrix(v[:3,:3]@T[0,:3,:3].T).magnitude()))
ids=np.arange(0,405,4)
for name,P in poses.items():
 ious=[];prec=[];rec=[]
 scene.set_pose(node,P)
 for i in ids:
  col,d=ren.render(scene,flags=RenderFlags.RGBA); a=col[:,:,3]>0 …[+482 chars]
```
<details><summary>output</summary>

```
A_pre pos [0.19028496 0.74740997 2.66871274] rotrel 0.7926809295525435
B_hold pos [0.18779321 0.75190157 2.70801497] rotrel 1.3955483320660638
all pos [0.19023123 0.74827361 2.67093325] rotrel 0.8185945851488468
base0 pos [0.19071217 0.73422575 2.63160014] rotrel 7.293956299530128e-15
base100 pos [0.19730757 0.74203068 2.65929747] rotrel 1.041045496869699
base350 pos [0.18711151 0.75393027 2.68535757] rotrel 1.3690053168954475
A_pre n 102 IoU mean/med 0.7448586861591292 0.7508102648018247 p10 0.6786518208902376 prec 0.9422395042049262 rec 0.7788540475292663
B_hold n 102 IoU mean/med 0.7651791674395796 0.773310367839265 p10 0.7077550345923989 prec 0.9746748231994183 rec 0.7803516996683884
all n 102 IoU mean/med 0.7458660319955461 0.7514306529498642 p10 0.6799603122748652 prec 0.943613927292 …[+427 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/maskdiff.py <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
for i in range(708):
 ok,im=cap.read(); frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
frames=np.array(frames)
h=h5py.File('/work/inputs/masks.h5','r'); o=h['object'][:]; p=h['person'][:]
# diff under object visible in both, eroded to avoid boundaries
for i in range(330,421):
 a=o[i]&~p[i]&o[i-1]&~p[i-1]
 a=cv2.erode(a.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
 if a.sum()>100:
  mad=np.mean(np.abs(frames[i].astype(float)-frames[i-1])[a]); frac=np.mean(np.abs(frames[i].astype(float)-frames[i-1])[a]>15)
 else: mad=frac=np.nan
 if i%2==0 or i in [373,374,375]: print(i,a.sum(),round(mad,2),round(frac,3))
PY
python /work/scratch/analysis/maskdiff.py"
```
<details><summary>output</summary>

```
330 11298 0.64 0.0
332 11625 0.69 0.0
334 11341 0.52 0.001
336 11164 0.66 0.0
338 11046 0.56 0.0
340 11012 0.95 0.0
342 11138 0.52 0.0
344 11065 0.74 0.0
346 11501 0.59 0.0
348 11580 0.68 0.001
350 11682 0.63 0.0
352 11989 0.56 0.0
354 11923 0.37 0.0
356 11837 0.53 0.001
358 11791 0.61 0.002
360 11670 0.86 0.002
362 12068 0.82 0.005
364 12019 0.89 0.003
366 12608 0.66 0.0
368 12950 0.6 0.0
370 13056 0.54 0.0
372 13108 0.63 0.0
373 13155 0.56 0.001
374 13208 0.6 0.001
375 13235 0.84 0.0
376 13275 0.66 0.0
378 13348 0.52 0.0
380 13427 0.76 0.0
382 13434 0.6 0.0
384 13358 0.62 0.001
386 13426 0.5 0.0
388 13346 0.56 0.0
390 13246 0.68 0.0
392 13187 0.52 0.0
394 13177 0.55 0.0
396 13147 0.72 0.0
398 13183 0.79 0.0
400 13101 1.76 0.005
402 13137 2.01 0.014
404 13146 2.2 0.025
406 13236 2.1 0.022 …[+142 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/ndr_test.py <<'PY'
import torch,numpy as np,trimesh,json,h5py
import nvdiffrast.torch as dr
dev='cuda'
W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'][0].astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(m.vertices,dtype=torch.float32,device=dev); F=torch.tensor(m.faces,dtype=torch.int32,device=dev)
Tm=torch.tensor(T,device=dev); Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev)
X=(Tm[:3,:3]@V.T+Tm[:3,3:4]); X=(Tcw[:3,:3]@X+Tcw[:3,3:4])
Kt=torch.tensor(K,dtype=torch.float32,device=dev); fx,fy=Kt[0,0],Kt[1,1]; cx,cy=Kt[0,2],Kt[1,2]
ctx=dr.RasterizeCudaContext(); h=h5py.File('/work/inputs/masks.h5','r'); b=h['object'][0]
for sy in [1,-1]:
 x=fx*X[0]+cx*X[2]; y=sy*(fy*X[1]+cy*X[2]); z=X[2]
 clip=torch.stack([x,y,z,X[2]],-1)[None]
 rast,_=dr.rasterize(ctx,clip,F,torch.tensor([H,W],device=dev))
 a=rast[0,:,:,3]>=0
 a=a.cpu().numpy()
 inter=(a&b).sum(); union=(a|b).sum(); print(sy,a.sum(),b.sum(),inter/union,inter/a.sum(),inter/b.sum(),np.where(a)[1].min(),np.where(a)[1].max(),np.where(a)[0].min(),np.where(a)[0].max())
PY
python /work/scratch/analysis/ndr_test.py"
```
<details><summary>output</summary>

```
1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151
-1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/ndr_test.py'; s=open(p).read(); s=s.replace(\"z=X[2]\\n clip=torch.stack([x,y,z,X[2]],-1)[None]\", \"near,far=.01,10.\\n z=((far+near)/(far-near))*X[2] - (2*far*near)/(far-near)\\n clip=torch.stack([x,y,z,X[2]],-1)[None]\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/ndr_test.py"
```
<details><summary>output</summary>

```
1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151
-1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/analysis/ndr_test.py').read().split(\"ctx=dr\")[0])
for sy in [1,-1]:
 xp=x/X[2]; yp=sy*y/X[2]; print(sy,xp.min().item(),xp.max().item(),yp.min().item(),yp.max().item(),X[2].min().item(),X[2].max().item(),(X[2]<=0).sum().item())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
NameError: name 'x' is not defined. Did you mean: 'X'?

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh,json
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0].astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(m.vertices); Tm=torch.tensor(T); Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32)
X=(Tm[:3,:3]@V.T+Tm[:3,3:4]); X=(Tcw[:3,:3]@X+Tcw[:3,3:4]); Kt=torch.tensor(K,dtype=torch.float32); fx,fy=Kt[0,0],Kt[1,1]; cx,cy=Kt[0,2],Kt[1,2]
for sy in [1,-1]:
 xp=(fx*X[0]+cx*X[2])/X[2]; yp=sy*(fy*X[1]+cy*X[2])/X[2]
 print(sy,xp.min().item(),xp.max().item(),yp.min().item(),yp.max().item(),X[2].min().item(),X[2].max().item())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
RuntimeError: expected m1 and m2 to have the same dtype, but got: float != double

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh,json
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0].astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(m.vertices,dtype=torch.float32); Tm=torch.tensor(T); Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32)
X=(Tm[:3,:3]@V.T+Tm[:3,3:4]); X=(Tcw[:3,:3]@X+Tcw[:3,3:4]); Kt=torch.tensor(K,dtype=torch.float32); fx,fy=Kt[0,0],Kt[1,1]; cx,cy=Kt[0,2],Kt[1,2]
for sy in [1,-1]:
 xp=(fx*X[0]+cx*X[2])/X[2]; yp=sy*(fy*X[1]+cy*X[2])/X[2]
 print(sy,xp.min().item(),xp.max().item(),yp.min().item(),yp.max().item(),X[2].min().item(),X[2].max().item())
PY"
```
<details><summary>output</summary>

```
1 670.0075073242188 861.526123046875 661.1217651367188 960.448974609375 2.346774101257324 3.0113446712493896
-1 670.0075073242188 861.526123046875 -960.448974609375 -661.1217651367188 2.346774101257324 3.0113446712493896

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/ndr_test.py'; s=open(p).read(); old=\"x=fx*X[0]+cx*X[2]; y=sy*(fy*X[1]+cy*X[2]); z=X[2]\\n near,far=.01,10.\\n z=((far+near)/(far-near))*X[2] - (2*far*near)/(far-near)\\n clip=torch.stack([x,y,z,X[2]],-1)[None]\"; new=\"x=(2*(fx*X[0]+cx*X[2])/W-1)*X[2]; y=sy*(2*(fy*X[1]+cy*X[2])/H-1)*X[2]\\n near,far=.01,10.\\n z=(((far+near)/(far-near))*X[2] - (2*far*near)/(far-near))\\n clip=torch.stack([x,y,z,X[2]],-1)[None]\"; assert old in s; open(p,'w').write(s.replace(old,new))
PY
python /work/scratch/analysis/ndr_test.py"
```
<details><summary>output</summary>

```
1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151
-1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,30p' /work/scratch/analysis/ndr_test.py"
```
<details><summary>output</summary>

```
import torch,numpy as np,trimesh,json,h5py
import nvdiffrast.torch as dr
dev='cuda'
W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'][0].astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(m.vertices,dtype=torch.float32,device=dev); F=torch.tensor(m.faces,dtype=torch.int32,device=dev)
Tm=torch.tensor(T,device=dev); Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev)
X=(Tm[:3,:3]@V.T+Tm[:3,3:4]); X=(Tcw[:3,:3]@X+Tcw[:3,3:4])
Kt=torch.tensor(K,dtype=torch.float32,device=dev); fx,fy=Kt[0,0],Kt[1,1]; cx,cy=Kt[0,2],Kt[1,2]
ctx=dr.RasterizeCudaContext(); h=h5py.File('/work/inputs/ma …[+501 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/ndr_test.py'; s=open(p).read(); s=s.replace('x=fx*X[0]+cx*X[2]; y=sy*(fy*X[1]+cy*X[2]); near,far=.01,10.', 'x=(2*(fx*X[0]+cx*X[2])/W-1)*X[2]; y=sy*(2*(fy*X[1]+cy*X[2])/H-1)*X[2]; near,far=.01,10.'); open(p,'w').write(s)
PY
python /work/scratch/analysis/ndr_test.py"
```
<details><summary>output</summary>

```
1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151
-1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/ndr_test.py'; s=open(p).read(); s=s.replace(\"x=(2*(fx*X[0]+cx*X[2])/W-1)*X[2]; y=sy*(2*(fy*X[1]+cy*X[2])/H-1)*X[2]; near,far=.01,10.\\n z=((far+near)/(far-near))*X[2] - (2*far*near)/(far-near)\\n clip=torch.stack([x,y,z,X[2]],-1)[None]\", \"x=2*(fx*X[0]+cx*X[2])/W-1; y=sy*(2*(fy*X[1]+cy*X[2])/H-1); near,far=.01,10.\\n z=(((far+near)/(far-near))*X[2] - (2*far*near)/(far-near))/X[2]\\n clip=torch.stack([x,y,z,torch.ones_like(z)],-1)[None]\"); open(p,'w').write(s)
PY
python /work/scratch/analysis/ndr_test.py"
```
<details><summary>output</summary>

```
1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151
-1 1769472 18224 0.010299117476851851 0.010299117476851851 1.0 0 1535 0 1151

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/analysis/ndr_test.py').read().split('for sy')[0])
# reproduce sy1
x=2*(fx*X[0]+cx*X[2])/W-1; y=2*(fy*X[1]+cy*X[2])/H-1; z=(((10.01)/(9.99))*X[2] - .02/9.99)/X[2]; clip=torch.stack([x,y,z,torch.ones_like(z)],-1)[None]; print(clip.min(1).values,clip.max(1).values)
rast,_=dr.rasterize(ctx,clip,F,torch.tensor([H,W],device=dev)); print(rast.shape,rast[...,3].min(),rast[...,3].max(),(rast[...,3]>=0).sum(),(rast[...,3]<0).sum(),torch.unique(rast[...,3])[:10])
PY"
```
<details><summary>output</summary>

```
tensor([[1.2870, 1.7714, 1.0011, 1.0000]], device='cuda:0') tensor([[2.0421, 3.5067, 1.0013, 1.0000]], device='cuda:0')
torch.Size([1, 1152, 1536, 4]) tensor(0., device='cuda:0') tensor(0., device='cuda:0') tensor(1769472, device='cuda:0') tensor(0, device='cuda:0') tensor([0.], device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/ndr_test.py'; s=open(p).read(); s=s.replace('x=2*(fx*X[0]+cx*X[2])/W-1; y=sy*(2*(fy*X[1]+cy*X[2])/H-1);', 'x=2*(fx*X[0]/X[2]+cx)/W-1; y=sy*(2*(fy*X[1]/X[2]+cy)/H-1);'); s=s.replace('a=rast[0,:,:,3]>=0','a=rast[0,:,:,3]>0'); open(p,'w').write(s)
PY
python /work/scratch/analysis/ndr_test.py"
```
<details><summary>output</summary>

```
1 15849 18224 0.7840201057646997 0.9447914694933435 0.8216637401229149 670 860 661 959
-1 15849 18224 0.0 0.0 0.0 670 860 192 490

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/analysis/ndr_test.py').read().split('for sy')[0])
import time
# fixed clip sy=1
x=2*(fx*X[0]/X[2]+cx)/W-1; y=2*(fy*X[1]/X[2]+cy)/H-1; z=(((10.01)/(9.99))*X[2] - .02/9.99)/X[2]; clip=torch.stack([x,y,z,torch.ones_like(z)],-1)[None]
for res in [[H,W],[576,768],[288,384]]:
 torch.cuda.synchronize(); t=time.time()
 for _ in range(20): rast,_=dr.rasterize(ctx,clip,F,torch.tensor(res,device=dev))
 torch.cuda.synchronize(); print(res,(time.time()-t)/20)
PY"
```
<details><summary>output</summary>

```
[1152, 1536] 0.00035517215728759767
[576, 768] 0.0002050161361694336
[288, 384] 0.00017431974411010742

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/ndr_tex.py <<'PY'
import torch,numpy as np,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0].astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(m.vertices,dtype=torch.float32,device=dev); F=torch.tensor(m.faces,dtype=torch.int32,device=dev); UV=torch.tensor(m.visual.uv,dtype=torch.float32,device=dev)
Tm=torch.tensor(T,device=dev); Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev)
X=(Tm[:3,:3]@V.T+Tm[:3,3:4]); X=(Tcw[:3,:3]@X+Tcw[:3,3:4]); Kt=torch.tensor(K,dtype=torch.float32,device=dev); fx,fy=Kt[0,0],Kt[1,1]; cx,cy=Kt[0,2],Kt[1,2]
x=2*(fx*X[0]/X[2]+cx)/W-1; y=2*(fy*X[1]/X[2]+cy)/H-1; z=(((10.01)/(9.99))*X[2] - .02/9.99)/X[2]; clip=torch.stack([x,y,z,torch.ones_like(z)],-1)[None]
ctx=dr.RasterizeCudaContext(); rast,_=dr.rasterize(ctx,clip,F,torch.tensor([H,W],device=dev)); attr,_=dr.interpolate(UV[None],rast,F); print(attr.shape,attr.min().item(),attr.max().item())
tex=np.array(m.visual.material.baseColorTexture); tex=torch.tensor(tex,dtype=torch.float32,device=dev).permute(2,0,1)[None]/255.
for flips in [0,1]:
 uv=attr.clone(); grid=uv*2-1
 if flips: grid=grid.flip(-1)
 if flips: grid[...,1]=-grid[...,1]
 col=torch.nn.functional.grid_sample(tex,grid,al …[+431 chars]
```
<details><summary>output</summary>

```
torch.Size([1, 1152, 1536, 2]) 0.0 0.0
Traceback (most recent call last):
  File "/work/scratch/analysis/ndr_tex.py", line 18, in <module>
    out=img.copy(); out[mask]=(col[mask][:,:3]*255).astype(np.uint8)[:,[2,1,0]]; cv2.imwrite(f'/work/scratch/analysis/ndrtex_{flips}.jpg',out)
        ^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'copy'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr, inspect
print(dr.interpolate.__doc__)
PY
grep -R \"dr.interpolate\" -n /opt/conda/lib/python3.11/site-packages/nvdiffrast 2>/dev/null | head"
```
<details><summary>output</summary>

```
Interpolate vertex attributes.

    All input tensors must be contiguous and reside in GPU memory. The output tensors
    will be contiguous and reside in GPU memory.

    Args:
        attr: Attribute tensor with dtype `torch.float32`. 
              Shape is [num_vertices, num_attributes] in range mode, or 
              [minibatch_size, num_vertices, num_attributes] in instanced mode.
              Broadcasting is supported along the minibatch axis.
        rast: Main output tensor from `rasterize()`.
        tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
        rast_db: (Optional) Tensor containing image-space derivatives of barycentrics, 
                 i.e., the second output tensor from `rasterize()`. Enables computing
                 image-space der …[+889 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/ndr_tex.py'; s=open(p).read(); s=s.replace(\"attr,_=dr.interpolate(UV[None],rast,F); print(attr.shape,attr.min().item(),attr.max().item())\", \"print('ids', (rast[...,3]>0).sum().item(),rast[...,3].max().item()); attr,_=dr.interpolate(UV[None].contiguous(),rast,F); print(attr.shape,attr.min().item(),attr.max().item(), (attr>0).sum().item())\"); s=s.replace(\"img=cv2.imread('/work/inputs/video.mp4');\", \"cap=cv2.VideoCapture('/work/inputs/video.mp4'); ok,img=cap.read();\"); open(p,'w').write(s)
PY
python /work/scratch/analysis/ndr_tex.py"
```
<details><summary>output</summary>

```
ids 0 0.0
torch.Size([1, 1152, 1536, 2]) 0.0 0.0 0
/work/scratch/analysis/ndr_tex.py:19: RuntimeWarning: Mean of empty slice.
  print(flips,col[mask].mean(0),col[mask].min(0),col[mask].max(0))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
Traceback (most recent call last):
  File "/work/scratch/analysis/ndr_tex.py", line 19, in <module>
    print(flips,col[mask].mean(0),col[mask].min(0),col[mask].max(0))
                                  ^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py", line 45, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: ze …[+67 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/clip=torch.stack(\\[x,y,z,torch.ones_like(z)\\],-1)\\[None\\]/clip=torch.stack([x,y,z,torch.ones_like(z)],-1)[None]; print('clip',clip.min(1).values,clip.max(1).values)/\" /work/scratch/analysis/ndr_tex.py && python /work/scratch/analysis/ndr_tex.py"
```
<details><summary>output</summary>

```
clip tensor([[-0.1276,  0.1478,  1.0011,  1.0000]], device='cuda:0') tensor([[0.1218, 0.6674, 1.0013, 1.0000]], device='cuda:0')
ids 0 0.0
torch.Size([1, 1152, 1536, 2]) 0.0 0.0 0
/work/scratch/analysis/ndr_tex.py:19: RuntimeWarning: Mean of empty slice.
  print(flips,col[mask].mean(0),col[mask].min(0),col[mask].max(0))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
Traceback (most recent call last):
  File "/work/scratch/analysis/ndr_tex.py", line 19, in <module>
    print(flips,col[mask].mean(0),col[mask].min(0),col[mask].max(0))
                                  ^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py", line 45, in _amin
    return umr_minimum( …[+196 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'x=\\|z=\\|clip=' /work/scratch/analysis/ndr_test.py"
```
<details><summary>output</summary>

```
10:ctx=dr.RasterizeCudaContext(); h=h5py.File('/work/inputs/masks.h5','r'); b=h['object'][0]
12: x=2*(fx*X[0]/X[2]+cx)/W-1; y=sy*(2*(fy*X[1]/X[2]+cy)/H-1); near,far=.01,10.
13: z=(((far+near)/(far-near))*X[2] - (2*far*near)/(far-near))/X[2]
14: clip=torch.stack([x,y,z,torch.ones_like(z)],-1)[None]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# instrument ndr_test context then invokes ctx same
s=open('/work/scratch/analysis/ndr_test.py').read(); s=s.replace(\"for sy in [1,-1]:\", \"print('clipctx',ctx)\\nfor sy in [1,-1]:\")
open('/work/scratch/analysis/ndr_test2.py','w').write(s)
PY
python /work/scratch/analysis/ndr_test2.py"
```
<details><summary>output</summary>

```
clipctx <nvdiffrast.torch.ops.RasterizeCudaContext object at 0x7dee0e266350>
1 15849 18224 0.7840201057646997 0.9447914694933435 0.8216637401229149 670 860 661 959
-1 15849 18224 0.0 0.0 0.0 670 860 192 490

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/analysis/ndr_test.py').read(); s=s.replace(\"clip=torch.stack([x,y,z,torch.ones_like(z)],-1)[None]\", \"clip=torch.stack([x,y,z,torch.ones_like(z)],-1)[None]; print('clip',clip.min(1).values,clip.max(1).values)\")
open('/work/scratch/analysis/ndr_test3.py','w').write(s)
PY
python /work/scratch/analysis/ndr_test3.py"
```
<details><summary>output</summary>

```
clip tensor([[-0.1276,  0.1478,  0.9935,  1.0000]], device='cuda:0') tensor([[0.1218, 0.6674, 0.9954, 1.0000]], device='cuda:0')
1 15849 18224 0.7840201057646997 0.9447914694933435 0.8216637401229149 670 860 661 959
clip tensor([[-0.1276, -0.6674,  0.9935,  1.0000]], device='cuda:0') tensor([[ 0.1218, -0.1478,  0.9954,  1.0000]], device='cuda:0')
-1 15849 18224 0.0 0.0 0.0 670 860 192 490

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/scratch/analysis/ndr_test3.py | sed -n '8,18p'; echo ---; nl -ba /work/scratch/analysis/ndr_tex.py | sed -n '7,16p'"
```
<details><summary>output</summary>

```
     8	X=(Tm[:3,:3]@V.T+Tm[:3,3:4]); X=(Tcw[:3,:3]@X+Tcw[:3,3:4])
     9	Kt=torch.tensor(K,dtype=torch.float32,device=dev); fx,fy=Kt[0,0],Kt[1,1]; cx,cy=Kt[0,2],Kt[1,2]
    10	ctx=dr.RasterizeCudaContext(); h=h5py.File('/work/inputs/masks.h5','r'); b=h['object'][0]
    11	for sy in [1,-1]:
    12	 x=2*(fx*X[0]/X[2]+cx)/W-1; y=sy*(2*(fy*X[1]/X[2]+cy)/H-1); near,far=.01,10.
    13	 z=(((far+near)/(far-near))*X[2] - (2*far*near)/(far-near))/X[2]
    14	 clip=torch.stack([x,y,z,torch.ones_like(z)],-1)[None]; print('clip',clip.min(1).values,clip.max(1).values)
    15	 rast,_=dr.rasterize(ctx,clip,F,torch.tensor([H,W],device=dev))
    16	 a=rast[0,:,:,3]>0
    17	 a=a.cpu().numpy()
    18	 inter=(a&b).sum(); union=(a|b).sum(); print(sy,a.sum(),b.sum(),inter/union,inter/a.sum(),inter/b.sum(),np.w …[+1179 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh,json
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0].astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(m.vertices,dtype=torch.float32,device=dev)
Tm=torch.tensor(T,device=dev); Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev)
X=(Tm[:3,:3]@V.T+Tm[:3,3:4]); X=(Tcw[:3,:3]@X+Tcw[:3,3:4]); print('X',X[2].min().item(),X[2].max().item())
near,far=.01,10.
z=(((far+near)/(far-near))*X[2] - (2*far*near)/(far-near))/X[2]
print('z',z.min().item(),z.max().item(),(z[0]).item())
z2=(((10.01)/(9.99))*X[2] - (.02/9.99))/X[2]
print('z2',z2.min().item(),z2.max().item())
PY"
```
<details><summary>output</summary>

```
X 2.346774101257324 3.0113446712493896
z 0.9934711456298828 0.9953537583351135 0.9945478439331055
z2 1.0011489391326904 1.0013371706008911

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
near,far=.01,10.
print(near,far,(far+near)/(far-near),(2*far*near)/(far-near))
PY"
```
<details><summary>output</summary>

```
0.01 10.0 1.002002002002002 0.02002002002002002

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's#z=(((10.01)/(9.99))\\*X\\[2\\] - .02/9.99)/X\\[2\\]#z=torch.full_like(X[2],0.5)#' /work/scratch/analysis/ndr_tex.py && python /work/scratch/analysis/ndr_tex.py"
```
<details><summary>output</summary>

```
clip tensor([[-0.1276,  0.1478,  0.5000,  1.0000]], device='cuda:0') tensor([[0.1218, 0.6674, 0.5000, 1.0000]], device='cuda:0')
ids 15849 105469.0
torch.Size([1, 1152, 1536, 2]) 0.0 0.9986968040466309 31698
0 [0.32354355 0.27187604 0.21786723] [0. 0. 0.] [0.90436304 0.8023607  0.6825054 ]
1 [0.3976248  0.31644738 0.23246127] [0. 0. 0.] [0.8813555  0.77945936 0.67989236]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/fit_static_ndr.py <<'PY'
import os,json,time,numpy as np,h5py,cv2,trimesh,torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
np.random.seed(0); torch.manual_seed(0)
dev='cuda'; W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K'],np.float32); Twc=np.array(c['T_world_cam'],np.float64)
r=np.load('/work/inputs/reconstruction.npz'); T0=r['T_world_mesh'].astype(np.float64)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();
V=torch.tensor(m.vertices,dtype=torch.float32,device=dev); F=torch.tensor(m.faces,dtype=torch.int32,device=dev)
Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev)
Kt=torch.tensor(K,device=dev); fx,fy=Kt[0,0],Kt[1,1]; cx,cy=Kt[0,2],Kt[1,2]
ctx=dr.RasterizeCudaContext(); res=torch.tensor([H,W],device=dev)
h=h5py.File('/work/inputs/masks.h5','r')
def prepare(ids):
    obj=h['object'][ids].astype(np.uint8); per=h['person'][ids].astype(np.uint8)
    # excludes uncertain person boundary but keeps visible object
    ker=np.ones((7,7),np.uint8)
    per=np.array([cv2.dilate(x,ker) for x in per])
    valid=(per==0)
    target=obj.astype(bool)&valid
    return torch.tensor(target,device=dev),torch.tensor(valid,device=dev),ids
def render_mask(P):
    Tm=torch.tensor(P,dtype=torch.float32,device=dev)
    X=Tm[:3,:3]@V.T+Tm[:3,3:4]; X=Tcw[:3,:3]@X+Tcw[:3,3:4]
    x=2*(fx*X[0]/X[2 …[+2181 chars]
```
<details><summary>output</summary>

```
A_pre 0.23621205985546112 sec 0.1758115291595459
B_hold 0.22053954005241394 sec 0.06690478324890137
base350 0.21699649095535278 sec 0.06666159629821777
base0 0.23883503675460815 sec 0.06740283966064453
base300 0.21095870435237885 sec 0.06668853759765625
Optimization terminated successfully.
         Current function value: 0.142952
         Iterations: 6
         Function evaluations: 576
opt 0.14295236766338348 [ 0.02451385  0.00820807 -0.00127574 -0.00673251 -0.01548906 -0.07822743] sec 80.88737154006958
P [[-0.73174349  0.68015667 -0.04402704  0.1810607 ]
 [-0.12866314 -0.0744099   0.9888928   0.73641251]
 [ 0.66932598  0.72928052  0.14196005  2.62978753]
 [ 0.          0.          0.          1.        ]]
final mean 0.18926401436328888
Optimization terminated successfully.
         Cur …[+481 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/depth_compare.py <<'PY'
import json,h5py,cv2,numpy as np,torch,trimesh
import nvdiffrast.torch as dr
dev='cuda'; W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K'],np.float32); Tcw=np.array(np.linalg.inv(np.array(c['T_world_cam'])),np.float32)
Kp=np.array(c['K_pipeline'],np.float32)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32); cp=r['contact_prob_right_left']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(m.vertices,dtype=torch.float32,device=dev); F=torch.tensor(m.faces,dtype=torch.int32,device=dev)
ctx=dr.RasterizeCudaContext(); res=torch.tensor([H,W],device=dev); TcwT=torch.tensor(Tcw,device=dev)
maskfile=h5py.File('/work/inputs/masks.h5','r'); OM=maskfile['object']; PM=maskfile['person']
depthfile=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
fit=np.load('/work/scratch/analysis/static_fit2.npz')
# candidates dict for given frame
def depth_for(P,Kuse):
 fx,fy=Kuse[0,0],Kuse[1,1]; cx,cy=Kuse[0,2],Kuse[1,2]
 Tm=torch.tensor(P,dtype=torch.float32,device=dev)
 X=Tm[:3,:3]@V.T+Tm[:3,3:4]; X=TcwT[:3,:3]@X+TcwT[:3,3:4]
 x=2*(fx*X[0]/X[2]+cx)/W-1; y=2*(fy*X[1]/X[2]+cy)/H-1
 clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None]
 rast,_=dr.rasterize(ctx,clip,F,res); alpha=rast[0,:,:,3]>0
 dep,_=dr.interpolate(X[2][None],rast,F)
 return alpha.cpu().numpy(),dep[0,:,:,0].cpu().numpy()
for i i …[+622 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/depth_compare.py", line 29, in <module>
    a,zr=depth_for(P,Kp)
         ^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/depth_compare.py", line 20, in depth_for
    dep,_=dr.interpolate(X[2][None],rast,F)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line  …[+716 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/X\\[2\\]\\[None\\],rast/X[2][None,:,None],rast/' /work/scratch/analysis/depth_compare.py && python /work/scratch/analysis/depth_compare.py"
```
<details><summary>output</summary>

```
0 [('base', 9662, 0.224, 0.23, 0.228), ('fitI', 10356, 0.225, 0.2207, 0.2309)]
100 [('base', 8745, 0.1881, 0.1911, 0.1965), ('fitI', 9751, 0.2209, 0.2162, 0.2283)]
200 [('base', 3908, 0.0158, 0.0705, 0.1755), ('fitI', 4266, 0.0961, 0.1385, 0.2043)]
300 [('base', 5113, -0.0477, -0.0656, 0.0816), ('fitI', 5334, 0.0178, 0.0026, 0.0679)]
350 [('base', 6888, -0.0276, -0.0153, 0.0737), ('fitI', 7358, 0.0337, 0.0435, 0.086)]
375 [('base', 8655, -0.0054, -0.0216, 0.0412), ('fitI', 8949, 0.0746, 0.0587, 0.0925)]
390 [('base', 8525, 0.0087, -0.0038, 0.0491), ('fitI', 8847, 0.0899, 0.0777, 0.1056)]
400 [('base', 8415, 0.0112, -0.0028, 0.0517)]
450 [('base', 9057, 0.0418, 0.0551, 0.056)]
500 [('base', 9099, 0.1379, 0.1819, 0.1409)]
550 [('base', 11364, 0.1706, 0.2221, 0.1707)]
600 [('base', 12406, 0.1 …[+281 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/.stages/05_object_pose.json && echo '---06---' && cat /pipeline/.stages/06_coconet.json && echo '---07---' && cat /pipeline/.stages/07_refinement.json"
```
<details><summary>output</summary>

```
{
  "name": "05_object_pose",
  "outputs": [
    {
      "mtime_ns": 1791207076870793329,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000008/foundationpose/episode_000008.pkl",
      "size": 203150
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py",
      "/home/hanzhou/Projects/v2d/run/out/episode_000008/export/episode_000008",
      "--depth-root",
      "/home/hanzhou/Projects/v2d/run/out/episode_000008/depth/moge2/aligned.h5",
      "--out-file",
      "/home/hanzhou/Projects/v2d/run/out/episode_000008/foundationpose/episode_000008.pkl",
      "--camera-id",
      "0",
      "--iteration",
      "5",
      "--max-attempts",
      "5",
 …[+5948 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/pipeline_report.json'
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
    "mtime_ns": 1791210872903778848,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000008/inference/episode_000008_step200000_before_after.mp4",
    "size": 50191861
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frame …[+5123 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"depth_correct\\|depth correction\\|T_world_mesh\\|depth_scale\" -n /code/reconstruction 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/mhr_input_renderer.py:49:    depth_scale = np.array([depth_m.shape[1] / rgb.shape[1], depth_m.shape[0] / rgb.shape[0]], dtype=np.float32)
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/mhr_input_renderer.py:50:    depth_center = center * depth_scale
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/mhr_input_renderer.py:51:    depth_crop_size = crop_size * float(depth_scale.mean())
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/mhr_input_renderer.py:53:    K_depth[0] *= depth_scale[0]
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/mhr_input_renderer.py:54:    K_depth[1] *= depth_scale[1]
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/mhr_input_renderer.py:55:    K_roi = _crop_intrinsics( …[+2338 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'grep -R "T_world_mesh" -n /code /workspace 2>/dev/null | head -100'
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); A=r['T_world_mesh'].astype(float); F=r['T_cam_mesh_foundationpose'].astype(float); C=r['T_cam_mesh_coconet'].astype(float); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Wf=Twc@F; Wc=Twc@C
for i in [0,100,300,350,374,400,450,500,550,600,650,700]:
 print('\\n',i,'A',np.round(A[i,:3,3],4),'Wf',np.round(Wf[i,:3,3],4),'Wc',np.round(Wc[i,:3,3],4))
 print('rot A-F',round(np.degrees(R.from_matrix(A[i,:3,:3].T@Wf[i,:3,:3]).magnitude()),2),'A-C',round(np.degrees(R.from_matrix(A[i,:3,:3].T@Wc[i,:3,:3]).magnitude()),2))
 print('cam z A', (np.linalg.inv(Twc)@A[i])[2,3], 'cf',F[i,2,3],'cc',C[i,2,3])
PY"
```
<details><summary>output</summary>

```

 0 A [0.1907 0.7342 2.6316] Wf [0.1541 0.7221 2.8035] Wc [0.1602 0.6857 2.6822]
rot A-F 0.5 A-C 1.43
cam z A 2.614588658496383 cf 2.785353660583496 cc 2.663363456726074

 100 A [0.1973 0.742  2.6593] Wf [0.1639 0.7161 2.7882] Wc [0.1667 0.6949 2.7181]
rot A-F 3.98 A-C 1.43
cam z A 2.6425952991755937 cf 2.770097255706787 cc 2.699601173400879

 300 A [0.1868 0.7504 2.69  ] Wf [0.1482 0.7047 2.7169] Wc [0.1541 0.6864 2.6847]
rot A-F 1.16 A-C 1.43
cam z A 2.6732524420983714 cf 2.6982815265655518 cc 2.6657657623291016

 350 A [0.1871 0.7539 2.6854] Wf [0.1472 0.7007 2.7051] Wc [0.1539 0.6854 2.6617]
rot A-F 1.61 A-C 1.43
cam z A 2.6687440628647248 cf 2.6863296031951904 cc 2.642780065536499

 374 A [0.1877 0.7515 2.7064] Wf [0.1535 0.6876 2.6559] Wc [0.1537 0.6779 2.6638]
rot A-F 1.74 A-C 1.43
 …[+1339 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
ims=[]
for f in ['ndrtex_0.jpg','ndrtex_1.jpg']:
 im=cv2.imread('/work/scratch/analysis/'+f); ims.append(cv2.resize(im[600:1020,500:1000],(500,420)))
cap=cv2.VideoCapture('/work/inputs/video.mp4');ok,im=cap.read(); ims.insert(0,cv2.resize(im[600:1020,500:1000],(500,420)))
cv2.imwrite('/work/scratch/analysis/tex_compare.jpg',np.hstack(ims))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/eval_static_detail.py <<'PY'
import os,json,numpy as np,h5py,cv2,trimesh,torch
import nvdiffrast.torch as dr
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32)
fit=np.load('/work/scratch/analysis/static_fit2.npz')
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev)
Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev)
h=h5py.File('/work/inputs/masks.h5','r'); OM=h['object']; PM=h['person']
def render(P):
 Tm=torch.tensor(P,dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4]
 x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None]
 r,_=dr.rasterize(ctx,clip,F,res);return r[0,:,:,3]>0
for phase,ids in [('initial',range(0,400,4)),('final',range(610,708,2))]:
 print('\\n',phase)
 for name,P in [('base0',T[0]),('base350',T[350]),('basefinal',T[690]),('fit',fit['initial' if phase=='initial' else 'final'])]:
  vals=[];aa=[];bb=[]
  for i in ids:
   per=cv2.dilate(PM[i].astype(np.uint8),np.ones((7,7),np.uint8)).asty …[+423 chars]
```
<details><summary>output</summary>

```

 initial
base0 iou 0.7612 prec 0.9394 rec 0.7994 med 0.7718
base350 iou 0.783 prec 0.9711 rec 0.8011 med 0.7905
basefinal iou 0.0049 prec 0.0138 rec 0.0117 med 0.0056
fit iou 0.857 prec 0.9942 rec 0.8613 med 0.8644

 final
base0 iou 0.0093 prec 0.0468 rec 0.0121 med 0.0085
base350 iou 0.0097 prec 0.0508 rec 0.0124 med 0.0088
basefinal iou 0.8112 prec 0.9618 rec 0.8382 med 0.8154
fit iou 0.8879 prec 0.9929 rec 0.8936 med 0.8894

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
h=h5py.File('/work/inputs/masks.h5','r'); o=h['object']; p=h['person']
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,100,300,350,390,500,600,700]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read()
 m=o[i].astype(np.uint8); n,lab,stats,c=cv2.connectedComponentsWithStats(m)
 print(i,'components',[(stats[j,4],stats[j,:4].tolist()) for j in range(1,n) if stats[j,4]>20])
 out=im.copy(); cont,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out,cont,-1,(0,0,255),2); cv2.imwrite(f'/work/scratch/analysis/mask_{i:03d}.jpg',out)
PY"
```
<details><summary>output</summary>

```
0 components [(18224, [664, 660, 197, 307])]
100 components [(14906, [719, 660, 143, 307])]
300 components [(10156, [663, 686, 147, 281])]
350 components [(1702, [814, 672, 24, 101]), (12053, [663, 678, 176, 289])]
390 components [(14747, [663, 669, 175, 298])]
500 components [(15059, [593, 651, 182, 274])]
600 components [(16745, [561, 651, 147, 255]), (26, [535, 842, 10, 3])]
700 components [(17587, [561, 651, 148, 255]), (26, [535, 842, 10, 3])]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/overlay_fits.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32);fit=np.load('/work/scratch/analysis/static_fit2.npz')
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev)
def render(P):
 Tm=torch.tensor(P,dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4];x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None];r,_=dr.rasterize(ctx,clip,F,res);return (r[0,:,:,3]>0).cpu().numpy()
cap=cv2.VideoCapture('/work/inputs/video.mp4');h=h5py.File('/work/inputs/masks.h5','r')
for i,ps in [(0,[('base0',T[0]),('fitI',fit['initial'])]),(200,[('base200',T[200]),('fitI',fit['initial'])]),(390,[('base390',T[390]),('fitI',fit['initial'])]),(620,[('base620',T[620]),('fitF',fit['final'])]),(700,[('base700',T[700]),('fitF',fit['final'])])]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read( …[+612 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# compute per-frame diff stats around final using cached less memory
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4'); G=[]
for _ in range(708): ok,im=cap.read(); G.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
h=h5py.File('/work/inputs/masks.h5','r');o=h['object'];p=h['person']
for i in range(580,708):
 a=o[i]&~p[i]&o[i-1]&~p[i-1];a=cv2.erode(a.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
 if a.sum()>100: mad=np.mean(np.abs(G[i].astype(float)-G[i-1])[a]); frac=np.mean(np.abs(G[i].astype(float)-G[i-1])[a]>15)
 else:mad=frac=np.nan
 print(i,a.sum(),round(mad,2),round(frac,3))
PY"
```
<details><summary>output</summary>

```
580 14788 2.69 0.025
581 14779 2.46 0.023
582 14753 2.09 0.011
583 14760 2.05 0.011
584 14675 2.22 0.017
585 14677 2.49 0.024
586 14625 2.17 0.016
587 14662 2.05 0.011
588 14667 2.0 0.009
589 14716 1.97 0.008
590 14827 2.2 0.016
591 14903 2.08 0.015
592 14802 2.19 0.015
593 14864 2.01 0.012
594 15095 1.68 0.002
595 14958 1.53 0.001
596 14959 0.74 0.0
597 14985 0.65 0.0
598 14988 0.89 0.0
599 15065 0.84 0.0
600 15087 0.97 0.0
601 15294 0.86 0.001
602 15102 1.22 0.001
603 15149 1.21 0.001
604 15303 1.2 0.002
605 15314 1.1 0.002
606 15356 1.0 0.001
607 15516 0.98 0.002
608 15689 1.01 0.002
609 15737 0.93 0.001
610 15814 1.15 0.001
611 15799 0.88 0.0
612 15792 0.83 0.001
613 15801 0.84 0.002
614 15765 1.04 0.002
615 15692 0.86 0.001
616 15572 0.73 0.002
617 15316 0.87 0.001
618 14989 0.87 0.00 …[+1685 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/eval_flips.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32);fit=np.load('/work/scratch/analysis/static_fit2.npz')
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r');OM=h['object'];PM=h['person']
def render(P):
 Tm=torch.tensor(P,dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4];x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None];r,_=dr.rasterize(ctx,clip,F,res);return (r[0,:,:,3]>0).cpu().numpy()
def score(P,ids):
 vals=[]
 for i in ids:
  per=cv2.dilate(PM[i].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);val=~per;tgt=OM[i]&val;a=render(P)&val;inter=(a&tgt).sum();vals.append(inter/((a|tgt).sum()))
 return np.mean(vals)
for phase,P,ids in [('initial',fit['initial' …[+603 chars]
```
<details><summary>output</summary>

```
initial
none 0.8567
x180 0.3381
y180 0.3532
z180 0.2865
final
none 0.8877
x180 0.4457
y180 0.4023
z180 0.6081
frame 0
none 0.8617
z180 0.2873
frame 400
none 0.8134
z180 0.3333
frame 450
none 0.7709
z180 0.2958
frame 500
none 0.7655
z180 0.3297
frame 550
none 0.8317
z180 0.4548
frame 595
none 0.8054
z180 0.6095
frame 650
none 0.8888
z180 0.6037

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);fit=np.load('/work/scratch/analysis/static_fit2.npz');P0=fit['initial'];P1=fit['final']
for i in [399,400,595,596]:
 B=T[i];D=P1@np.linalg.inv(B) if i>=596 else P0@np.linalg.inv(B)
 print(i,'B',np.round(B[:3,3],4),'Dtrans',np.round(D[:3,3],4),'Drot',np.round(R.from_matrix(D[:3,:3]).as_rotvec()*180/np.pi,2))
PY"
```
<details><summary>output</summary>

```
399 B [0.1878 0.7516 2.7071] Dtrans [-0.0299  0.0516 -0.0933] Drot [ 1.4   0.47 -0.07]
400 B [0.1855 0.7514 2.7033] Dtrans [-0.0232  0.0543 -0.0904] Drot [1.47 0.4  0.02]
595 B [-0.3244  0.7378  2.9927] Dtrans [1.9682 0.2186 0.1511] Drot [  4.38 -29.85   0.31]
596 B [-0.3261  0.7386  2.9883] Dtrans [-0.0689  0.1397 -0.0926] Drot [2.8  1.76 1.27]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);f=np.load('/work/scratch/analysis/static_fit2.npz')
for key,i in [('initial',0),('initial',399),('final',596),('final',650),('final',700)]:
 P=f[key]; print(key,i,'brel',np.degrees(R.from_matrix(P[:3,:3]@T[i,:3,:3].T).magnitude()),'rotvec',np.round(R.from_matrix(P[:3,:3]@T[i,:3,:3].T).as_rotvec()*180/np.pi,2),'dt',np.round(P[:3,3]-T[i,:3,3],4))
PY"
```
<details><summary>output</summary>

```
initial 0 brel 1.8257472057023079 rotvec [ 1.58 -0.89  0.16] dt [-0.0097  0.0022 -0.0018]
initial 399 brel 1.4829854435527334 rotvec [ 1.4   0.47 -0.07] dt [-0.0067 -0.0152 -0.0773]
final 596 brel 3.54364526677342 rotvec [2.8  1.76 1.27] dt [ 0.0088 -0.0139 -0.0514]
final 650 brel 2.2260707948193246 rotvec [0.97 1.89 0.66] dt [ 0.0014 -0.0067 -0.0523]
final 700 brel 2.2260707948193246 rotvec [0.97 1.89 0.66] dt [ 0.0017 -0.0073 -0.0547]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for i in range(585,601):
 rd=np.degrees(R.from_matrix(T[i,:3,:3]@T[i-1,:3,:3].T).magnitude()); print(i,np.round(T[i,:3,3],4),round(rd,2))
print('595-600 rel',np.degrees(R.from_matrix(T[600,:3,:3]@T[595,:3,:3].T).magnitude()))
PY"
```
<details><summary>output</summary>

```
585 [-0.315   0.7353  2.9884] 0.09
586 [-0.316   0.7324  2.9818] 0.39
587 [-0.3156  0.7346  2.9843] 0.69
588 [-0.3201  0.735   2.9859] 0.38
589 [-0.3198  0.7379  2.9874] 0.11
590 [-0.3231  0.741   2.995 ] 0.78
591 [-0.3225  0.7396  2.9911] 0.59
592 [-0.3238  0.7347  2.9889] 0.44
593 [-0.3238  0.7352  2.982 ] 0.66
594 [-0.3266  0.7384  2.99  ] 0.61
595 [-0.3244  0.7378  2.9927] 0.26
596 [-0.3261  0.7386  2.9883] 0.16
597 [-0.3255  0.7374  2.9855] 0.66
598 [-0.3258  0.7375  2.9899] 0.38
599 [-0.3287  0.738   2.9903] 0.38
600 [-0.3265  0.7397  2.9918] 0.42
595-600 rel 0.4989725575962126

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);f=np.load('/work/scratch/analysis/static_fit2.npz');P=f['final'];B=T[595]
D=P@np.linalg.inv(B)
print('Rprod\\n',P[:3,:3]@B[:3,:3].T)
print('Drot\\n',D[:3,:3]); print('rel',R.from_matrix(P[:3,:3]@B[:3,:3].T).as_rotvec()*180/np.pi);print('Drel',R.from_matrix(D[:3,:3]).as_rotvec()*180/np.pi)
print('P euler',R.from_matrix(P[:3,:3]).as_euler('xyz',degrees=True));print('B euler',R.from_matrix(B[:3,:3]).as_euler('xyz',degrees=True))
PY"
```
<details><summary>output</summary>

```
Rprod
 [[ 0.99928882 -0.02343898  0.02950676]
 [ 0.02483507  0.99854078 -0.04785353]
 [-0.02834113  0.04855264  0.99841874]]
Drot
 [[ 0.99929065 -0.02343971  0.02950574]
 [ 0.02483468  0.99855034 -0.04785408]
 [-0.02834222  0.04855257  0.99841816]]
rel [2.76356601 1.65825663 1.38381857]
Drel [2.76356601 1.65825663 1.38381857]
P euler [  66.12013274  -72.91985843 -156.89308518]
B euler [  76.5766927   -72.16228677 -168.25482469]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);f=np.load('/work/scratch/analysis/static_fit2.npz')
for name,key,i in [('d0','initial',399),('d1','final',596)]:
 D=f[key]@np.linalg.inv(T[i]); print(name,'t',np.round(D[:3,3],4),'rv',np.round(R.from_matrix(D[:3,:3]).as_rotvec()*180/np.pi,2))
PY"
```
<details><summary>output</summary>

```
d0 t [-0.0299  0.0516 -0.0933] rv [ 1.4   0.47 -0.07]
d1 t [-0.0689  0.1397 -0.0926] rv [2.8  1.76 1.27]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/dynamic_candidates.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);fit=np.load('/work/scratch/analysis/static_fit2.npz');P0=fit['initial'];P1=fit['final']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r');OM=h['object'];PM=h['person']
# endpoint residual corrections
Rc0=P0[:3,:3]@T[399,:3,:3].T; Rc1=P1[:3,:3]@T[596,:3,:3].T
t0=P0[:3,3]-T[399,:3,3];t1=P1[:3,3]-T[596,:3,3]
q0=R.from_matrix(Rc0);q1=R.from_matrix(Rc1)
print('corr0',R.from_matrix(Rc0).as_rotvec()*180/np.pi,t0);print('corr1',R.from_matrix(Rc1).as_rotvec()*180/np.pi,t1)
# smooth baseline rotations/translations 390:607
a,b=390,607; idx=np.arange(a,b); rawr=R.from_matrix(T[idx,:3,:3]); q=rawr.as_quat();
# align quaternion signs
for j in range(1,len(q)):
 if np.dot(q[j],q[j-1])<0:q[j]*=- …[+2184 chars]
```
<details><summary>output</summary>

```
corr0 [ 1.40453989  0.47028791 -0.07309456] [-0.00669528 -0.01523663 -0.07730237]
corr1 [2.80304408 1.75731754 1.26972467] [ 0.00875365 -0.01391462 -0.05135106]
baseline IoU 0.7972 med 0.7947 prec 0.9615 rec 0.8232 p10 0.7669
corr_sm0 IoU 0.8539 med 0.8541 prec 0.9742 rec 0.8734 p10 0.8246
corr_sm5 IoU 0.8552 med 0.8588 prec 0.9749 rec 0.8742 p10 0.8246
corr_sm9 IoU 0.8559 med 0.8586 prec 0.9757 rec 0.8744 p10 0.8263
corr_sm13 IoU 0.856 med 0.8565 prec 0.9756 rec 0.8746 p10 0.8253
corr_sm17 IoU 0.8563 med 0.8563 prec 0.9758 rec 0.8747 p10 0.8284
corr_sm21 IoU 0.856 med 0.8594 prec 0.9757 rec 0.8745 p10 0.826
sm5 IoU 0.7975 med 0.7952 prec 0.962 rec 0.8233 p10 0.7713
sm9 IoU 0.7968 med 0.7957 prec 0.9617 rec 0.8227 p10 0.7722
sm13 IoU 0.7974 med 0.7976 prec 0.9619 rec 0.8232 p10 0.7693
sm17 …[+57 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/centroid_eval.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);P=np.load('/work/scratch/analysis/dyn_default.npz')['T'].astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r');OM=h['object'];PM=h['person']
def render(i):
 Tm=torch.tensor(P[i],dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4];x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None];r,_=dr.rasterize(ctx,clip,F,res);return (r[0,:,:,3]>0).cpu().numpy()
resids=[]
for i in range(400,596):
 a=render(i); per=cv2.dilate(PM[i].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);val=~per;tgt=OM[i]&val;a=a&val
 ya,xa=np.where(a);yt,xt=np.where(tgt)
 if len(xa)>100 and len(xt)>100:
  du=xt.mean()-xa.mean();dv=yt.mean()-ya.mean(); resids.append([i,du,dv,len(xa),len(xt)])
resids=np.array(resids)
print('mean du,dv',resids[:,1:3].mean(0) …[+217 chars]
```
<details><summary>output</summary>

```
mean du,dv [-2.20692587  7.8808242 ] median [-2.80963691  7.92047928] absmean [2.32691264 7.8808242 ] p90 abs [3.50849413 9.25912658]
[ 4.00000000e+02 -2.42996237e+00  6.55224287e+00  1.28210000e+04
  1.42650000e+04]
[ 4.20000000e+02 -3.68486967e+00  6.59565835e+00  1.28110000e+04
  1.44810000e+04]
[ 4.40000000e+02 -2.05638100e+00  6.73361704e+00  1.37900000e+04
  1.51430000e+04]
[ 4.60000000e+02 -3.22342721e+00  6.83903619e+00  1.34720000e+04
  1.47220000e+04]
[ 4.80000000e+02 -1.68117483e+00  9.15724412e+00  1.29130000e+04
  1.45470000e+04]
[ 5.00000000e+02 -3.51296411e+00  8.52110095e+00  1.33110000e+04
  1.47580000e+04]
[ 5.20000000e+02 -3.03007575e+00  7.82420093e+00  1.35140000e+04
  1.50660000e+04]
[ 5.40000000e+02 -2.97150941e+00  7.86668792e+00  1.39580000e+04
  1.59240000e+04]
[5 …[+154 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# quick modify centroid_eval to static
s=open('/work/scratch/analysis/centroid_eval.py').read(); s=s.replace(\"for i in range(400,596):\",\"for i in list(range(0,400,4))+list(range(596,708,4)):\")
open('/work/scratch/analysis/centroid_eval_static.py','w').write(s)
PY
python /work/scratch/analysis/centroid_eval_static.py"
```
<details><summary>output</summary>

```
mean du,dv [-0.92928297  7.08806029] median [-1.30652561  6.62257544] absmean [1.69589455 7.08806029] p90 abs [2.90070276 8.72531754]
[ 0.00000000e+00 -6.44765120e-01  6.18596529e+00  1.58310000e+04
  1.82240000e+04]
[8.00000000e+01 4.85193873e-01 4.57051885e+00 1.56690000e+04
 1.75600000e+04]
[ 160.           -5.44372933   16.87048995 3855.         4995.        ]
[ 2.40000000e+02 -1.52457105e+00  6.81014244e+00  1.51710000e+04
  1.73970000e+04]
[ 3.20000000e+02 -2.24999978e+00  5.80747710e+00  9.48600000e+03
  1.08450000e+04]
[5.96000000e+02 3.28509284e-01 7.78996899e+00 1.41930000e+04
 1.59900000e+04]
[6.76000000e+02 1.15299547e+00 7.02201484e+00 1.59090000e+04
 1.76590000e+04]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);f=np.load('/work/scratch/analysis/static_fit2.npz');P0=f['initial'];P1=f['final'];g=json.load(open('/work/inputs/ground_plane.json'));n=np.array(g['plane'][:3]);d=g['plane'][3];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.c_[m.vertices,np.ones(len(m.vertices))]
for name,P,ids in [('base0',T[0],[0]),('fitI',P0,[0,399]),('base596',T[596],[596]),('base700',T[700],[700]),('fitF',P1,[596,700])]:
 print(name)
 for i in ids:
  w=(P@v.T).T[:,:3];sd=w@n+d;print(i,'min',sd.min(),'p1',np.percentile(sd,1),'p5',np.percentile(sd,5))
PY"
```
<details><summary>output</summary>

```
base0
0 min 0.0031719285747173753 p1 0.045066735503749754 p5 0.18430822020141074
fitI
0 min -0.005642354404904459 p1 0.04499349745980652 p5 0.18364326082242388
399 min -0.005642354404904459 p1 0.04499349745980652 p5 0.18364326082242388
base596
596 min -0.02187729337210964 p1 0.027470543442291717 p5 0.16449245447883099
base700
700 min -0.006836834464967634 p1 0.03497202293543502 p5 0.17294091242689064
fitF
596 min 0.008196775092537134 p1 0.046098150136290796 p5 0.1844591740387529
700 min 0.008196775092537134 p1 0.046098150136290796 p5 0.1844591740387529

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/contact_eval.py <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
r=np.load('/work/inputs/reconstruction.npz');B=r['T_world_mesh'].astype(float);C=np.load('/work/scratch/analysis/dyn_default.npz')['T'].astype(float);kp=r['human_keypoints_world'];cp=r['contact_prob_right_left']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.c_[m.vertices,np.ones(len(m.vertices))]
for name,T in [('base',B),('corr',C)]:
 ds=[]
 for i in range(400,596):
  if cp[i].max()>.5:
   w=(T[i]@v.T).T[:,:3];tree=cKDTree(w)
   hands=np.r_[kp[i,21:41],kp[i,42:62]]
   d,ix=tree.query(hands)
   ds.append([i,cp[i,0],cp[i,1],d.min(),np.percentile(d,5),d.mean()])
 ds=np.array(ds)
 print(name,'n',len(ds),'mindist mean/med/p90',np.mean(ds[:,3]),np.median(ds[:,3]),np.percentile(ds[:,3],90),'p5mean',np.mean(ds[:,4]),'allmean',np.mean(ds[:,5]))
# compare per-frame difference
PY
python /work/scratch/analysis/contact_eval.py"
```
<details><summary>output</summary>

```
base n 196 mindist mean/med/p90 0.0026686044078560274 0.002185753416635292 0.005530789340618339 p5mean 0.005455496504338598 allmean 0.036430334349384134
corr n 196 mindist mean/med/p90 0.0020852650159899106 0.0019663064209729424 0.0029712545526830838 p5mean 0.004065375937989498 allmean 0.03196966465062019

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/rot_candidates.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);r=np.load('/work/inputs/reconstruction.npz');B=r['T_world_mesh'].astype(float);C=np.load('/work/scratch/analysis/dyn_default.npz')['T'].astype(float);F=r['T_cam_mesh_foundationpose'].astype(float);CC=r['T_cam_mesh_coconet'].astype(float);Wf=Twc@F;Wc=Twc@CC
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);Faces=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r');OM=h['object'];PM=h['person']
def render(P):
 Tm=torch.tensor(P,dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4];x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None];rr,_=dr.rasterize(ctx,clip,Faces,res);return (rr[0,:,:,3]>0).cpu().numpy()
def score(P,ids):
 vals=[]
 for i in ids:
  per=cv2.dilate(PM[i].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);val=~per;tgt=OM[i]&val; …[+542 chars]
```
<details><summary>output</summary>

```
TypeError: float() argument must be a string or a real number, not 'scipy.spatial.transform._rotation.Rotation'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/work/scratch/analysis/rot_candidates.py", line 21, in <module>
    q0=R.from_matrix(C[i,:3,:3]);q1=R.from_matrix(Wr[i,:3,:3]);P2[i,:3,:3]=q0.inv()*q1 # not used
                                                               ~~^^^^^^^^^
ValueError: setting an array element with a sequence.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/rot_candidates.py';s=open(p).read();s=s.replace(\" # rotations slerp half\\n P2=C.copy();\\n for i in range(400,596):\\n  q0=R.from_matrix(C[i,:3,:3]);q1=R.from_matrix(Wr[i,:3,:3]);P2[i,:3,:3]=q0.inv()*q1 # not used\\n cands[nm+'_half']=P2\\n\",\"\\n\");open(p,'w').write(s)
PY
python /work/scratch/analysis/rot_candidates.py"
```
<details><summary>output</summary>

```
base 0.7972
corr 0.8559
fp 0.8392
coco 0.8539

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
P=np.load('/work/scratch/analysis/dyn_default.npz')['T']
for i in list(range(394,407))+list(range(589,602)):
 if i==0:continue
 rd=np.degrees(R.from_matrix(P[i,:3,:3]@P[i-1,:3,:3].T).magnitude());td=np.linalg.norm(P[i,:3,3]-P[i-1,:3,3]); print(i,np.round(P[i,:3,3],4),round(td,4),round(rd,2))
PY"
```
<details><summary>output</summary>

```
394 [0.1811 0.7364 2.6298] 0.0 0.0
395 [0.1811 0.7364 2.6298] 0.0 0.0
396 [0.1811 0.7364 2.6298] 0.0 0.0
397 [0.1811 0.7364 2.6298] 0.0 0.0
398 [0.1811 0.7364 2.6298] 0.0 0.0
399 [0.1811 0.7364 2.6298] 0.0 0.0
400 [0.1788 0.7354 2.6291] 0.0026 0.14
401 [0.1777 0.7349 2.6274] 0.0021 0.08
402 [0.1765 0.7345 2.6276] 0.0013 0.11
403 [0.1754 0.7344 2.6278] 0.0011 0.06
404 [0.1746 0.7347 2.6291] 0.0015 0.1
405 [0.1734 0.7358 2.6329] 0.0041 0.07
406 [0.172  0.7358 2.6351] 0.0026 0.12
589 [-0.3118  0.7237  2.9377] 0.003 0.28
590 [-0.313   0.7242  2.9386] 0.0016 0.21
591 [-0.3145  0.7241  2.938 ] 0.0016 0.16
592 [-0.315   0.7238  2.9381] 0.0006 0.22
593 [-0.3159  0.7232  2.9378] 0.001 0.17
594 [-0.3162  0.7229  2.9363] 0.0016 0.22
595 [-0.3165  0.7233  2.9366] 0.0006 0.15
596 [-0.3173  0.7247  2.93 …[+206 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/fit_static_weighted.py <<'PY'
import os,json,time,numpy as np,h5py,cv2,trimesh,torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
np.random.seed(0);torch.manual_seed(0)
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);T0=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
prev=np.load('/work/scratch/analysis/static_fit2.npz')
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);Kt=torch.tensor(K,device=dev);fx,fy=Kt[0,0],Kt[1,1];cx,cy=Kt[0,2],Kt[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev)
h=h5py.File('/work/inputs/masks.h5','r')
def prep(ids):
 obj=h['object'][ids].astype(np.uint8);per=h['person'][ids].astype(np.uint8);ker=np.ones((7,7),np.uint8);per=np.array([cv2.dilate(x,ker) for x in per]);valid=per==0;tgt=obj.astype(bool)&valid
 # weight by visible area, clipped so very occluded frames still contribute modestly
 w=np.sqrt(np.maximum(tgt.sum((1,2)),1).astype(np.float32));w=w/w.sum()
 return torch.tensor(tgt,device=dev),torch.tensor(valid,device=dev),torch.tensor(w,device=dev)
def rend(P):
 Tm=torch.tensor(P,dtype= …[+1283 chars]
```
<details><summary>output</summary>

```
initial start 0.13743872940540314
Optimization terminated successfully.
         Current function value: 0.107152
         Iterations: 6
         Function evaluations: 719
initial opt 0.10715197026729584 [-0.00833914 -0.00172969 -0.01041943 -0.00397892 -0.01730868 -0.05861736] sec 58.629013538360596
final start 0.11202333867549896
Optimization terminated successfully.
         Current function value: 0.103810
         Iterations: 5
         Function evaluations: 544
final opt 0.10381026566028595 [ 0.01078762 -0.00504645 -0.00514387  0.00553753 -0.0090481  -0.03897629] sec 26.757650136947632
[[-0.73417277  0.6781132  -0.03395343  0.17708178]
 [-0.11544537 -0.07539704  0.99044821  0.71910384]
 [ 0.66907602  0.73107987  0.13363948  2.57117017]
 [ 0.          0.          0.          1.         …[+272 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
u=np.load('/work/scratch/analysis/static_fit2.npz');w=np.load('/work/scratch/analysis/static_fit_weighted.npz');g=json.load(open('/work/inputs/ground_plane.json'));n=np.array(g['plane'][:3]);d=g['plane'][3];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.c_[m.vertices,np.ones(len(m.vertices))]
for name,P in [('uI',u['initial']),('wI',w['initial']),('uF',u['final']),('wF',w['final'])]:
 ww=(P@v.T).T[:,:3];sd=ww@n+d;print(name,'min',sd.min(),'p0.1',np.percentile(sd,.1),'p1',np.percentile(sd,1),'trans',P[:3,3])
PY"
```
<details><summary>output</summary>

```
uI min -0.005642354404904459 p0.1 0.0031604678544499536 p1 0.04499349745980652 trans [0.1810607  0.73641251 2.62978753]
wI min 0.01672556437074202 p0.1 0.02554873397205954 p1 0.06470195016694298 trans [0.17708178 0.71910384 2.57117017]
uF min 0.008196775092537134 p0.1 0.012091264990659045 p1 0.046098150136290796 trans [-0.31731055  0.72471126  2.93697048]
wF min 0.016862814907113055 p0.1 0.023243485281485787 p1 0.05802781452656802 trans [-0.31177302  0.71566316  2.89799419]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
r=np.load('/work/inputs/reconstruction.npz');B=r['T_world_mesh'].astype(float);C=np.load('/work/scratch/analysis/dyn_default.npz')['T'];g=json.load(open('/work/inputs/ground_plane.json'));n=np.array(g['plane'][:3]);d=g['plane'][3];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.c_[m.vertices,np.ones(len(m.vertices))]
for name,T in [('base',B),('corr',C)]:
 mins=[]; p1s=[]
 for i in range(400,596):
  w=(T[i]@v.T).T[:,:3];sd=w@n+d;mins.append(sd.min());p1s.append(np.percentile(sd,.1))
 print(name,'min plane pct',np.percentile(mins,[0,10,50,90,100]),'p0.1',np.percentile(p1s,[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
base min plane pct [-0.02872158 -0.01861787 -0.01018123 -0.00191895  0.00570581] p0.1 [-0.02206334 -0.01155355 -0.00301474  0.00447221  0.01190179]
corr min plane pct [-0.00924673 -0.00451459  0.00281285  0.01087257  0.01637491] p0.1 [-0.00049922  0.00428428  0.0116502   0.01948925  0.02313596]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/corr_interp.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.signal import savgol_filter
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);fit=np.load('/work/scratch/analysis/static_fit2.npz');P0=fit['initial'];P1=fit['final']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r');OM=h['object'];PM=h['person']
Rc0=P0[:3,:3]@B[399,:3,:3].T;Rc1=P1[:3,:3]@B[596,:3,:3].T;t0=P0[:3,3]-B[399,:3,3];t1=P1[:3,3]-B[596,:3,3];q0=R.from_matrix(Rc0);q1=R.from_matrix(Rc1);sp=Slerp([0,1],R.concatenate([q0,q1]))
idx=np.arange(390,607);q=R.from_matrix(B[idx,:3,:3]).as_quat()
for j in range(1,len(q)):
 if np.dot(q[j],q[j-1])<0:q[j]*=-1
q=savgol_filter(q,17,2,axis=0,mode='interp');q/=np.linalg.norm(q,axis=1,keepdims=True);RR=R.from_quat(q).as_matrix();tt=savgol_filter(B[idx,:3,3],17,2,axis=0,mode='interp')
def build(mode):
 out=B.copy();ou …[+1246 chars]
```
<details><summary>output</summary>

```
smooth 0.85629 0.85631 0.97576 0.87472
linear 0.85553 0.85463 0.97524 0.8744
const 0.84073 0.8402 0.96625 0.86619
early 0.83948 0.83339 0.96711 0.8639

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/time_metrics.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);P=np.load('/work/scratch/analysis/dyn_smooth.npz')['T'].astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r');OM=h['object'];PM=h['person']
def render(T):
 Tm=torch.tensor(T,dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4];x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None];r,_=dr.rasterize(ctx,clip,F,res);return (r[0,:,:,3]>0).cpu().numpy()
for i in range(400,596,10):
 per=cv2.dilate(PM[i].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);val=~per;tgt=OM[i]&val;a=render(P[i])&val;inter=(a&tgt).sum();iou=inter/((a|tgt).sum());prec=inter/a.sum();rec=inter/tgt.sum();print(i,round(iou,3),round(prec,3),round(rec,3),a.sum(),tgt.sum())
PY
python /work/scratch/analysis/time_metrics.py"
```
<details><summary>output</summary>

```
400 0.891 0.994 0.895 12846 14265
410 0.884 0.991 0.891 13071 14531
420 0.879 0.995 0.883 12857 14481
430 0.879 0.979 0.897 13848 15115
440 0.887 0.986 0.898 13780 15143
450 0.867 0.973 0.888 14128 15485
460 0.869 0.973 0.891 13483 14722
470 0.858 0.974 0.878 13032 14469
480 0.828 0.963 0.855 12923 14547
490 0.831 0.974 0.85 13018 14920
500 0.829 0.955 0.863 13336 14758
510 0.829 0.975 0.847 13164 15163
520 0.831 0.961 0.86 13481 15066
530 0.827 0.964 0.853 13760 15556
540 0.828 0.97 0.85 13956 15924
550 0.858 0.981 0.872 14323 16113
560 0.852 0.976 0.871 14894 16680
570 0.854 0.978 0.871 14784 16593
580 0.873 0.992 0.879 14390 16240
590 0.875 0.99 0.882 14486 16258

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/analysis/overlay_fits.py').read()
# switch to dynamic file and frames
s=s.replace(\"fit=np.load('/work/scratch/analysis/static_fit2.npz')\",\"fit=np.load('/work/scratch/analysis/dyn_smooth.npz')\")
start=s.index('for i,ps in')
s=s[:start]+'''for i in [400,440,480,500,520,550,590]:\\n cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();P=fit['T']\\n a=render(P[i]).astype(np.uint8);cont,_=cv2.findContours(a,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cont,-1,(255,0,255),2,cv2.LINE_AA)\\n om=h['object'][i].astype(np.uint8);cont,_=cv2.findContours(om,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cont,-1,(0,255,0),1,cv2.LINE_AA)\\n cv2.putText(im,str(i),(700,1050),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)\\n cv2.imwrite(f'/work/scratch/analysis/dynoverlay_{i:03d}.jpg',im)\\n'''
open('/work/scratch/analysis/overlay_dyn.py','w').write(s)
PY
python /work/scratch/analysis/overlay_dyn.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,glob
fs=sorted(glob.glob('/work/scratch/analysis/dynoverlay_*.jpg'));ims=[]
for f in fs:
 im=cv2.imread(f);ims.append(cv2.resize(im[620:1020,500:900],(400,400)))
cv2.imwrite('/work/scratch/analysis/dynoverlay_montage.jpg',np.vstack([np.hstack(ims[i:i+3]) for i in range(0,6,3)]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/fit_initial_early.py <<'PY'
exec(open('/work/scratch/analysis/fit_static_ndr.py').read().split(\"ids=np.arange(0,400,4)\")[0])
prev=np.load('/work/scratch/analysis/static_fit2.npz')['initial']
for name,ids in [('early',np.arange(0,347,3)),('early2',np.arange(0,347,2))]:
 target,valid,ids=prepare(ids);P0=prev;fn=make_obj(target,valid);print(name,'start',fn(np.zeros(6)));import time;t=time.time();o=minimize(fn,np.zeros(6),method='Powell',bounds=[(-.12,.12)]*3+[(-.06,.06)]*3,options={'maxiter':80,'xtol':1e-5,'ftol':1e-7,'disp':True});print(name,o.fun,o.x,time.time()-t);P=np.eye(4);P[:3,:3]=R.from_rotvec(o.x[:3]).as_matrix()@P0[:3,:3];P[:3,3]=P0[:3,3]+o.x[3:];np.save('/work/scratch/analysis/early_'+name+'.npy',P);print(P)
PY
python /work/scratch/analysis/fit_initial_early.py"
```
<details><summary>output</summary>

```
early start 0.14696474373340607
Optimization terminated successfully.
         Current function value: 0.112177
         Iterations: 8
         Function evaluations: 1067
early 0.11217749118804932 [-0.01313879 -0.00029989 -0.00607237 -0.00334572 -0.01763059 -0.05908109] 97.38107800483704
[[-0.7326855   0.67950252 -0.03805647  0.17771498]
 [-0.1154134  -0.0689486   0.99092171  0.71878192]
 [ 0.67070986  0.73042619  0.12894134  2.57070644]
 [ 0.          0.          0.          1.        ]]
early2 start 0.14702849090099335
Traceback (most recent call last):
                                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/ …[+1283 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/depth_grid.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);P0=np.load('/work/scratch/analysis/static_fit2.npz')['initial']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r');OM=h['object'];PM=h['person']
ids=np.arange(0,347,3)
def rend(P):
 Tm=torch.tensor(P,dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4];x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None];r,_=dr.rasterize(ctx,clip,F,res);return (r[0,:,:,3]>0).cpu().numpy()
ray=Twc[:3,:3]@np.array([0,0,1.])
for er in [0,1,2]:
 print('erode',er)
 for dz in [-.08,-.06,-.04,-.02,0,.02,.04,.06]:
  P=P0.copy();P[:3,3]+=dz*ray;vals=[]
  for i in ids:
   obj=OM[i].astype(np.uint8)
   if er:obj=cv2.erode(obj,np.ones((2*er+1,2*er+1),np.uint8))
   per=cv2.dilate(PM[i].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);val=~per;tgt=(obj>0)&val;a=rend(P) …[+176 chars]
```
<details><summary>output</summary>

```
erode 0
-0.08 0.77878; -0.06 0.80663; -0.04 0.83255; -0.02 0.85154; 0 0.85304; 0.02 0.83728; 0.04 0.81195; 0.06 0.78543; 
erode 1
-0.08 0.77807; -0.06 0.81513; -0.04 0.85203; -0.02 0.88464; 0 0.90388; 0.02 0.89195; 0.04 0.86552; 0.06 0.83589; 
erode 2
-0.08 0.74888; -0.06 0.78528; -0.04 0.82272; -0.02 0.85567; 0 0.87611; 0.02 0.88333; 0.04 0.8653; 0.06 0.83569; 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/fit_eroded.py <<'PY'
import os,json,time,numpy as np,h5py,cv2,trimesh,torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);prev=np.load('/work/scratch/analysis/static_fit2.npz')
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r')
def prep(ids,er=1):
 obj=h['object'][ids].astype(np.uint8)
 if er:obj=np.array([cv2.erode(x,np.ones((2*er+1,2*er+1),np.uint8)) for x in obj])
 per=h['person'][ids].astype(np.uint8);per=np.array([cv2.dilate(x,np.ones((7,7),np.uint8)) for x in per]);valid=per==0;tgt=(obj>0)&valid
 return torch.tensor(tgt,device=dev),torch.tensor(valid,device=dev)
def rend(P):
 Tm=torch.tensor(P,dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4];x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None];r,_=dr.rasterize(ctx,clip,F,res);return r[0,:,:,3]>0
def obj(tgt,val, …[+891 chars]
```
<details><summary>output</summary>

```
initial start 0.09682712703943253
Optimization terminated successfully.
         Current function value: 0.077581
         Iterations: 6
         Function evaluations: 651
initial 0.07758123427629471 [-0.0152423   0.0001111  -0.00676518 -0.00254267 -0.00468388 -0.01208405] 32.23021602630615
final start 0.09031394124031067
Optimization terminated successfully.
         Current function value: 0.074850
         Iterations: 3
         Function evaluations: 291
final 0.0748499408364296 [ 0.02421374 -0.00263534  0.00025342  0.00022511 -0.00336249 -0.00245532] 7.186187028884888
[[-0.73248815  0.67975641 -0.03731405  0.17851803]
 [-0.11349314 -0.06788618  0.99121681  0.73172863]
 [ 0.67125287  0.73028946  0.12687351  2.61770348]
 [ 0.          0.          0.          1.        ]]
[[-0.27262807  0 …[+189 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
u=np.load('/work/scratch/analysis/static_fit2.npz');e=np.load('/work/scratch/analysis/static_eroded.npz');g=json.load(open('/work/inputs/ground_plane.json'));n=np.array(g['plane'][:3]);d=g['plane'][3];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.c_[m.vertices,np.ones(len(m.vertices))]
for name,P in [('uI',u['initial']),('eI',e['initial']),('uF',u['final']),('eF',e['final'])]:
 w=(P@v.T).T[:,:3];sd=w@n+d;print(name,'min',sd.min(),'p0.1',np.percentile(sd,.1),'t',P[:3,3])
PY"
```
<details><summary>output</summary>

```
uI min -0.005642354404904459 p0.1 0.0031604678544499536 t [0.1810607  0.73641251 2.62978753]
eI min 0.0033383268461122384 p0.1 0.012202024002956854 t [0.17851803 0.73172863 2.61770348]
uF min 0.008196775092537134 p0.1 0.012091264990659045 t [-0.31731055  0.72471126  2.93697048]
eF min 0.007348849087557063 p0.1 0.015046369156755357 t [-0.31708544  0.72134877  2.93451515]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed "s#static_fit2.npz#static_eroded.npz#; s#dyn_smooth.npz#dyn_eroded_smooth.npz#" /work/scratch/analysis/corr_interp.py > /work/scratch/analysis/corr_eroded.py
# Need labels dict and builds all; run
python /work/scratch/analysis/corr_eroded.py'
```
<details><summary>output</summary>

```
smooth 0.85848 0.85499 0.97821 0.875
linear 0.85864 0.85885 0.97796 0.87544
const 0.83913 0.83658 0.96438 0.86595
early 0.8397 0.83494 0.96819 0.86325

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed 's#dyn_default.npz#dyn_eroded_linear.npz#' /work/scratch/analysis/contact_eval.py > /work/scratch/analysis/contact_eroded.py && python /work/scratch/analysis/contact_eroded.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/contact_eroded.py", line 3, in <module>
    r=np.load('/work/inputs/reconstruction.npz');B=r['T_world_mesh'].astype(float);C=np.load('/work/scratch/analysis/dyn_eroded_linear.npz')['T'].astype(float);kp=r['human_keypoints_world'];cp=r['contact_prob_right_left']
                                                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/analysis/dyn_eroded_linear.npz'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's#dyn_default.npz#dyn_linear.npz#' /work/scratch/analysis/contact_eval.py > /work/scratch/analysis/contact_eroded.py && python /work/scratch/analysis/contact_eroded.py"
```
<details><summary>output</summary>

```
base n 196 mindist mean/med/p90 0.0026686044078560274 0.002185753416635292 0.005530789340618339 p5mean 0.005455496504338598 allmean 0.036430334349384134
corr n 196 mindist mean/med/p90 0.002345056423705046 0.0019682383527773197 0.0031701100173495526 p5mean 0.004920333792574163 allmean 0.03491936035796542

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's#dyn_default.npz#dyn_smooth.npz#' /work/scratch/analysis/contact_eval.py > /work/scratch/analysis/contact_eroded_smooth.py && python /work/scratch/analysis/contact_eroded_smooth.py"
```
<details><summary>output</summary>

```
base n 196 mindist mean/med/p90 0.0026686044078560274 0.002185753416635292 0.005530789340618339 p5mean 0.005455496504338598 allmean 0.036430334349384134
corr n 196 mindist mean/med/p90 0.0024043129283507904 0.0020696004660663214 0.0031543826591231847 p5mean 0.004856157608951165 allmean 0.03514413891518608

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
P=np.load('/work/scratch/analysis/dyn_smooth.npz')['T'];e=np.load('/work/scratch/analysis/static_eroded.npz')
print(np.max(abs(P[0]-e['initial'])),np.max(abs(P[399]-e['initial'])),np.max(abs(P[596]-e['final'])),np.max(abs(P[707]-e['final'])))
np.savez('/work/scratch/analysis/final_candidate.npz',T_world_mesh=P.astype(np.float64))
PY"
```
<details><summary>output</summary>

```
0.0 0.0 0.0 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/validate_final.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);r=np.load('/work/inputs/reconstruction.npz');B=r['T_world_mesh'].astype(np.float32);P=np.load('/work/scratch/analysis/final_candidate.npz')['T_world_mesh'].astype(np.float32);kp=r['human_keypoints_world'];cp=r['contact_prob_right_left']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r');OM=h['object'];PM=h['person']
def render(T):
 Tm=torch.tensor(T,dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4];x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None];rr,_=dr.rasterize(ctx,clip,F,res);return (rr[0,:,:,3]>0).cpu().numpy()
def metrics(T):
 mus=[];mds=[]
 for i in range(len(T)):
  per=cv2.dilate(PM[i].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);val=~per;tgt=OM[i]&val;a=render(T[i]) …[+2828 chars]
```
<details><summary>output</summary>

```
{
  "untouched_static_0_346": {
    "base_iou_mean": 0.7648002035470729,
    "base_iou_median": 0.7870544090056285,
    "result_iou_mean": 0.8638384267758866,
    "result_iou_median": 0.8738866728594066,
    "base_boundary_px": 2.0377193308014006,
    "result_boundary_px": 0.9789132931046939
  },
  "grasp_static_347_399": {
    "base_iou_mean": 0.7936825020048504,
    "base_iou_median": 0.7990153682389728,
    "result_iou_mean": 0.88351874050156,
    "result_iou_median": 0.8818657071084256,
    "base_boundary_px": 1.774500552213417,
    "result_boundary_px": 0.8307330394690892
  },
  "push_400_595": {
    "base_iou_mean": 0.796533595554841,
    "base_iou_median": 0.7970933555415893,
    "result_iou_mean": 0.8579621764019406,
    "result_iou_median": 0.855033601315788,
    "base_boundary_px …[+2023 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/fit_direct_eroded.py <<'PY'
exec(open('/work/scratch/analysis/fit_eroded.py').read().split(\"P0=go(\")[0])
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
from scipy.spatial.transform import Rotation as RR
P0=np.eye(4);P0[:3,:3]=RR.from_matrix(T[:347,:3,:3]).mean().as_matrix();P0[:3,3]=np.median(T[:347,:3,3],axis=0)
P1=np.eye(4);P1[:3,:3]=RR.from_matrix(T[640:708,:3,:3]).mean().as_matrix();P1[:3,3]=np.median(T[640:708,:3,3],axis=0)
print('starts',P0[:3,3],P1[:3,3])
# use larger bounds version
def go2(P0,ids,nm):
 tgt,val=prep(ids);f=obj(tgt,val,P0);print(nm,'start',f(np.zeros(6)));t=time.time();o=minimize(f,np.zeros(6),method='Powell',bounds=[(-.15,.15)]*3+[(-.08,.08)]*3,options={'maxiter':100,'xtol':1e-5,'ftol':1e-7,'disp':True});print(nm,o.fun,o.x,time.time()-t);P=np.eye(4);P[:3,:3]=R.from_rotvec(o.x[:3]).as_matrix()@P0[:3,:3];P[:3,3]=P0[:3,3]+o.x[3:];return P
PI=go2(P0,np.arange(0,347,6),'initial');PF=go2(P1,np.arange(610,708,3),'final');np.savez('/work/scratch/analysis/static_direct_eroded.npz',initial=PI,final=PF);print(PI);print(PF)
PY
python /work/scratch/analysis/fit_direct_eroded.py"
```
<details><summary>output</summary>

```
starts [0.19032006 0.74678898 2.66455913] [-0.31937572  0.73255375  2.99383414]
initial start 0.22313140332698822
Optimization terminated successfully.
         Current function value: 0.086589
         Iterations: 9
         Function evaluations: 1104
initial 0.08658857643604279 [ 0.01226107 -0.0030636   0.00311932 -0.01055277 -0.01712143 -0.05713682] 50.36658835411072
final start 0.18219107389450073
Optimization terminated successfully.
         Current function value: 0.079698
         Iterations: 6
         Function evaluations: 669
final 0.07969817519187927 [ 0.03974818  0.02921995  0.01009725  0.00018274 -0.00973627 -0.05443422] 17.29194450378418
[[-0.73094639  0.68091924 -0.04545728  0.17976729]
 [-0.11490995 -0.05714572  0.99173085  0.72966755]
 [ 0.67269093  0.73012557  0.12001485 …[+335 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF chair trajectory.

The released baseline is a useful per-frame estimate but it jitters while the
chair is at rest and accumulates a depth offset.  This script:

1. detects the two long static object phases from the interaction timeline,
2. fits one pooled pose to each static phase with a differentiable silhouette
   rasterizer (one-pixel-eroded SAM masks are used for the final boundary fit,
   which compensates for the mask'"'s outward boundary bias),
3. smooths the tracked dynamic segment and transfers the endpoint static-pose
   corrections smoothly through it.

Only the files under --inputs and the installed software are used.  Runtime is
approximately 3-4 minutes on the provided GPU.
\"\"\"
import argparse
import json
import os
import time

os.environ.setdefault(\"PYOPENGL_PLATFORM\", \"egl\")

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot, Slerp

# Object phases verified from the video, object-mask motion, and contact
# probabilities.  The chair is stationary through the grasp, starts sliding at
# frame 400, stops by frame 596, and remains stationary while the person leaves.
INITIAL_END = 399       # last frame of the initial static phase
DYNAMIC_START = 400     # first frame …[+8356 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
[phase] fitting initial static pose
[fit] initial/raw: start loss=0.220540
[fit] initial/raw: end loss=0.141291, dt=[-0.00785365 -0.01626296 -0.07994082], dr=[ 0.00917047  0.00406808 -0.00419031], 59.7s
[fit] initial/eroded: start loss=0.086000
[fit] initial/eroded: end loss=0.075991, dt=[-0.00105497 -0.00450785 -0.00955553], dr=[-0.0024426   0.00277659 -0.0034625 ], 19.1s
[phase] fitting final static pose
[fit] final/raw: start loss=0.189264
[fit] final/raw: end loss=0.108129, dt=[ 0.00351864 -0.01194463 -0.07364243], dr=[0.0172952  0.02823961 0.00457827], 60.0s
[fit] final/eroded: start loss=0.091011
[fit] final/eroded: end loss=0.072032, dt=[ 0.0009323  -0.00453234 -0.00234676], dr=[0.03349054 0.00162804 0.00755684], 10.5s
[done] wrote /work/outputs/object_poses_world.npz (708 frames) i …[+9 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
P=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];C=np.load('/work/scratch/analysis/final_candidate.npz')['T_world_mesh'];F=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
print(P.shape,P.dtype,np.isfinite(P).all())
for i in [0,100,300,399,400,450,500,550,595,596,650,707]:
 print(i,'out',np.round(P[i,:3,3],4),'cand',np.round(C[i,:3,3],4),'base',np.round(F[i,:3,3],4),'out-cand deg',round(np.degrees(R.from_matrix(P[i,:3,:3]@C[i,:3,:3].T).magnitude()),2))
print('static diff t max',np.max(np.abs(P[0]-C[0])),np.max(np.abs(P[596]-C[596])))
PY"
```
<details><summary>output</summary>

```
(708, 4, 4) float64 True
0 out [0.1789 0.7311 2.6185] cand [0.1785 0.7317 2.6177] base [0.1907 0.7342 2.6316] out-cand deg 0.17
100 out [0.1789 0.7311 2.6185] cand [0.1785 0.7317 2.6177] base [0.1973 0.742  2.6593] out-cand deg 0.17
300 out [0.1789 0.7311 2.6185] cand [0.1785 0.7317 2.6177] base [0.1868 0.7504 2.69  ] out-cand deg 0.17
399 out [0.1789 0.7311 2.6185] cand [0.1785 0.7317 2.6177] base [0.1878 0.7516 2.7071] out-cand deg 0.17
400 out [0.1768 0.7301 2.6163] cand [0.1764 0.7307 2.6155] base [0.1855 0.7514 2.7033] out-cand deg 0.17
450 out [0.0627 0.7165 2.6981] cand [0.062  0.7179 2.7002] base [0.0681 0.7394 2.7903] out-cand deg 0.08
500 out [-0.0921  0.7143  2.8335] cand [-0.0934  0.7174  2.8417] base [-0.0941  0.7371  2.9168] out-cand deg 0.22
550 out [-0.2118  0.7074  2.8457] …[+598 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's#/work/scratch/analysis/final_candidate.npz#/work/outputs/object_poses_world.npz#' /work/scratch/analysis/validate_final.py > /work/scratch/analysis/validate_output.py && python /work/scratch/analysis/validate_output.py"
```
<details><summary>output</summary>

```
{
  "untouched_static_0_346": {
    "base_iou_mean": 0.7648002035470729,
    "base_iou_median": 0.7870544090056285,
    "result_iou_mean": 0.8644191284027741,
    "result_iou_median": 0.8740103270223752,
    "base_boundary_px": 2.0377193308014006,
    "result_boundary_px": 0.9750225918437287
  },
  "grasp_static_347_399": {
    "base_iou_mean": 0.7936825020048504,
    "base_iou_median": 0.7990153682389728,
    "result_iou_mean": 0.8825483863222869,
    "result_iou_median": 0.8804402201100551,
    "base_boundary_px": 1.774500552213417,
    "result_boundary_px": 0.8383577004918512
  },
  "push_400_595": {
    "base_iou_mean": 0.796533595554841,
    "base_iou_median": 0.7970933555415893,
    "result_iou_mean": 0.8556059153340263,
    "result_iou_median": 0.854480602882697,
    "base_boundary_ …[+2020 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/flips_output.py <<'PY'
import json,numpy as np,cv2,trimesh,torch,h5py
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
dev='cuda';W,H=1536,1152
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Twc=np.array(c['T_world_cam']);P=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev);Tcw=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=dev);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ctx=dr.RasterizeCudaContext();res=torch.tensor([H,W],device=dev);h=h5py.File('/work/inputs/masks.h5','r');OM=h['object'];PM=h['person']
def render(T):
 Tm=torch.tensor(T,dtype=torch.float32,device=dev);X=Tm[:3,:3]@V.T+Tm[:3,3:4];X=Tcw[:3,:3]@X+Tcw[:3,3:4];x=2*(fx*X[0]/X[2]+cx)/W-1;y=2*(fy*X[1]/X[2]+cy)/H-1;clip=torch.stack([x,y,torch.full_like(x,.5),torch.ones_like(x)],-1)[None];r,_=dr.rasterize(ctx,clip,F,res);return (r[0,:,:,3]>0).cpu().numpy()
def score(P0,ids):
 v=[]
 for i in ids:
  per=cv2.dilate(PM[i].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);val=~per;tgt=OM[i]&val;a=render(P0)&val;inter=(a&tgt).sum();v.append(inter/((a|tgt).sum()))
 return np.mean(v)
for nm,ids in [('initial',range(0,400,8)),('dynamic',range(400,596,8)),('final',range(596,708,4))] …[+327 chars]
```
<details><summary>output</summary>

```
initial
identity 0.8675
x180 0.3345
y180 0.3595
z180 0.2831
dynamic
identity 0.8567
x180 0.3897
y180 0.404
z180 0.3857
final
identity 0.8859
x180 0.4599
y180 0.4141
z180 0.5998

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy,scipy,cv2,h5py,trimesh,torch,nvdiffrast,pyrender
mods=[('numpy',numpy),('scipy',scipy),('opencv-python',cv2),('h5py',h5py),('trimesh',trimesh),('torch',torch),('nvdiffrast',nvdiffrast),('pyrender',pyrender)]
for n,m in mods: print(n,getattr(m,'__version__','unknown'))
import sys; print('python',sys.version.split()[0])
PY
git -C /code/reconstruction rev-parse HEAD 2>/dev/null || true"
```
<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
opencv-python 4.11.0
h5py 3.16.0
trimesh 5.1.1
torch 2.5.1+cu124
nvdiffrast 0.4.0
pyrender 0.1.45
python 3.11.10
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import PIL
print(PIL.__version__)
PY"
```
<details><summary>output</summary>

```
10.2.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# 6-DoF Skinny Wood Chair Trajectory

## Result

"'`object_poses_world.npz` contains `T_world_mesh` as a `(708, 4, 4)` float64 array. The mapping is the mesh-file frame to the released world frame, using OpenCV camera/world conventions.

The final trajectory uses:

- **Exactly one constant world pose for frames 0--399** (untouched approach + grasp).
- **A smoothed, endpoint-corrected rigid trajectory for frames 400--595** (chair slides toward/under the round table).
- **Exactly one constant world pose for frames 596--707** (chair released and person walks away).

The deterministic entry point is:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

Measured wall/GPU runtime for the final run: **195.1 s**.

## Phases found

| Frames | Object phase | Evidence | Confidence |
|---|---|---|---|
| 0--346 | Initial rest, person approaches | Chair mask/bbox and background pixels are static while person moves; contact probability is zero | High |
| 347--399 | Initial rest, person grasps | Contact rises to ~1.0, but frame-to-frame differences on visible chair pixels remain at background-noise level; tracker had a 3.0 cm depth jump at f374 while the image did not move | High |
| 400--595 | Chair slides/pushes | Sustained visible-chair image difference begins at f400; smooth world translation and about 29 deg of yaw accumulate | Medium-high |
| 596--707 | Final rest after releas …[+6679 chars]
```
<details><summary>output</summary>

```
total 120
-rw-rw-r-- 1 hanzhou hanzhou  6447 Oct 20 21:06 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou  1604 Oct 20 21:06 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 20:59 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 90902 Oct 20 21:02 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  9614 Oct 20 20:59 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, os, shutil
cache='/work/outputs/__pycache__'
if os.path.isdir(cache): shutil.rmtree(cache)
p='/work/outputs/object_poses_world.npz'
a=np.load(p)
assert a.files==['T_world_mesh'], a.files
T=a['T_world_mesh']
assert T.shape==(708,4,4) and T.dtype==np.float64 and np.isfinite(T).all()
assert np.allclose(T[:,3,:],np.array([0,0,0,1.],dtype=T.dtype))
print('OK',p,T.shape,T.dtype,os.path.getsize(p))
print('static initial max diff',np.max(np.abs(T[:400]-T[0])))
print('static final max diff',np.max(np.abs(T[596:]-T[596])))
PY
sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py /work/outputs/REPORT.md /work/outputs/TOOLS.md
find /work/outputs -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort"
```
<details><summary>output</summary>

```
OK /work/outputs/object_poses_world.npz (708, 4, 4) float64 90902
static initial max diff 0.0
static final max diff 0.0
f3dc3e4fe5d3a2b1a431295823285a475f39740b8c7c011a9ef1c6a14c77cdf6  /work/outputs/object_poses_world.npz
0224669e507151df0602c863e59db5afcb69a48c7bae69a1541931fe932da441  /work/outputs/solve.py
52ff67c7ebc9c5d91dd6270bc1734e3eb430ad406536aa04adb1407d172d2cad  /work/outputs/REPORT.md
a2408535c90f7bb8a5cc9c788a9f50cee774aba04a32706e9bee4298817bfe8a  /work/outputs/TOOLS.md
REPORT.md 6447 bytes
TOOLS.md 1604 bytes
object_poses_world.npz 90902 bytes
solve.py 9614 bytes

```
</details>

**Agent:** Completed the task and produced all required outputs:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(708, 4, 4)`, float64
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, method, validation, confidence, runtime
- `/work/outputs/TOOLS.md` — dependencies and tools

Final solver runtime: **195.1 seconds**. The trajectory uses pooled constant poses for static frames `0–399` and `596–707`, with a smoothed, endpoint-corrected motion trajectory for frames `400–595`. Mean visible-mask IoU improved from `0.76–0.82` baseline to `0.86–0.89` across phases; 180° symmetry hypotheses were explicitly tested and rejected.
