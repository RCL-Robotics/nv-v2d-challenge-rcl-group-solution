# T1_ds_ep005

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T16:02:45Z

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
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '35,100p'"
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
    42	4. **Global decoding over time**: choose one hyp …[+5588 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%f %s bytes\\n' | sort && echo '---episode---' && cat /work/inputs/episode.json && echo '---task---' && cat /work/inputs/task.txt && echo '---camera---' && cat /work/inputs/camera.json && echo '---ground---' && cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
camera.json 827 bytes
episode.json 68 bytes
frame0_reference_human.npz 192935 bytes
ground_plane.json 399 bytes
masks.h5 8166311 bytes
object_mesh.glb 19197156 bytes
reconstruction.npz 50516207 bytes
task.txt 69 bytes
video.mp4 31632938 bytes
---episode---
{
 "episode_index": 5,
 "object": "big_black_box",
 "frames": 1003
}---task---
Walk over to the G1 Box and rotate the box around the bounding area.
---camera---
{
 "K": [
  [
   819.9998779296875,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.9998779296875,
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
   834.7869873046875,
   0.0,
   768.0
  ],
  [
   0.0,
   834.7869873046875,
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
   0.9995985309959748,
   0.00490953441784938,
   0.0279047 …[+857 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -100; echo '--- code ---'; find /code/reconstruction -maxdepth 2 -type f | head -50; echo '--- gpu ---'; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1848
/pipeline/.stages/01_depth.json 1745
/pipeline/.stages/02_export.json 1968
/pipeline/.stages/03_human_initialization.json 2149
/pipeline/.stages/05_object_pose.json 1773
/pipeline/.stages/06_coconet.json 2829
/pipeline/.stages/07_refinement.json 2133
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/depth_report.json 1815
/pipeline/depth/moge2/aligned.h5 1222922860
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 106729
/pipeline/depth/moge2/raw.npy 3549560960
/pipeline/export/episode_000005/edex 563
/pipeline/export/episode_000005/wild_export.json 2801
/pipeline/foundationpose/episode_000005.pkl 284878
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 76702 …[+2231 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, numpy as np, json
p='/work/inputs/reconstruction.npz'
z=np.load(p)
print(z.files)
for k in z.files:
 print(k,z[k].shape,z[k].dtype)
with h5py.File('/work/inputs/masks.h5','r') as f:
 print('masks',f.keys())
 for k in f.keys(): print(k,f[k].shape,f[k].dtype)
PY"
```
<details><summary>output</summary>

```
['T_world_mesh', 'T_cam_mesh_foundationpose', 'T_cam_mesh_coconet', 'human_vertices_world', 'human_keypoints_world', 'contact_prob_right_left']
T_world_mesh (1003, 4, 4) float32
T_cam_mesh_foundationpose (1003, 4, 4) float32
T_cam_mesh_coconet (1003, 4, 4) float32
human_vertices_world (1003, 4610, 3) float32
human_keypoints_world (1003, 70, 3) float32
contact_prob_right_left (1003, 2) float32
masks <KeysViewHDF5 ['object', 'person']>
object (1003, 1152, 1536) bool
person (1003, 1152, 1536) bool

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, cv2, trimesh
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh']; F=len(T)
print('video',cv2.VideoCapture('/work/inputs/video.mp4').get(cv2.CAP_PROP_FRAME_COUNT),cv2.VideoCapture('/work/inputs/video.mp4').get(cv2.CAP_PROP_FPS))
# rot relative adjacent
R=T[:,:3,:3]; t=T[:,:3,3]
angs=[]
for i in range(1,F):
 d=R[i].T@R[i-1]
 ang=np.degrees(np.arccos(np.clip((np.trace(d)-1)/2,-1,1)))
 angs.append(ang)
angs=np.array(angs)
dts=np.linalg.norm(t[1:]-t[:-1],axis=1)
print('t range',t.min(0),t.max(0));print('rotang mean med max p95',angs.mean(),np.median(angs),angs.max(),np.percentile(angs,95)); print('dt mean med max p95',dts.mean(),np.median(dts),dts.max(),np.percentile(dts,95))
print('jumps top')
for i in np.argsort(angs)[-30:][::-1]: print(i,i+1,round(angs[i],2),round(dts[i],4),t[i],t[i+1])
# static candidate segments via t and R
for start in range(0,F,50):
 sl=slice(start,min(F,start+50)); print(start, 'tmean',np.round(t[sl].mean(0),3), 'tdisp',np.round(np.linalg.norm(t[sl]-t[sl].mean(0),axis=1).max(),3), 'rmax',round(angs[start:min(F-1,start+49)].max() if start<F-1 else 0,1))
# mesh
geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(geo), len(geo.vertices),len(geo.faces),geo.bounds,geo.extents,geo.centroid)
PY"
```
<details><summary>output</summary>

```
video 1003.0 30.0
t range [-0.12542744  0.7494865   2.3890934 ] [0.61477196 0.99688387 3.3881698 ]
rotang mean med max p95 0.740725389857613 0.5688950325944055 4.945616744377101 1.7486628624942004
dt mean med max p95 0.014139825 0.009395371 0.10852503 0.04362873174250125
jumps top
863 864 4.95 0.0129 [0.46208608 0.7657069  2.6249769 ] [0.4536239 0.7611813 2.6164057]
522 523 3.83 0.0205 [0.41728777 0.88004154 3.0868006 ] [0.4272392 0.8838726 3.1043088]
524 525 3.32 0.036 [0.4239039  0.86685026 2.997179  ] [0.43402678 0.87698317 3.030249  ]
523 524 2.93 0.1085 [0.4272392 0.8838726 3.1043088] [0.4239039  0.86685026 2.997179  ]
814 815 2.86 0.0148 [0.5459995 0.8441277 2.8241668] [0.5493503  0.84537095 2.8384993 ]
520 521 2.85 0.0365 [0.40099582 0.86759317 3.0480304 ] [0.41052225 0.875134   3.0 …[+3444 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/frames && ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -vf \"fps=1,scale=384:288\" /work/scratch/frames/f_%03d.jpg && python - <<'PY'
from PIL import Image,ImageDraw
import glob,os
files=glob.glob('/work/scratch/frames/*.jpg')
# 5x5 chunks
for ci in range(0,len(files),25):
 fs=files[ci:ci+25]; sh=Image.new('RGB',(384*5,288*5),'white');d=ImageDraw.Draw(sh)
 for j,f in enumerate(fs):
  im=Image.open(f); sh.paste(im,((j%5)*384,(j//5)*288)); d.text(((j%5)*384+5,(j//5)*288+5),os.path.basename(f),fill='red',stroke_width=2,stroke_fill='white')
 sh.save(f'/work/scratch/contact_{ci//25:02d}.jpg',quality=90)
print(len(files))
PY
ls -lh /work/scratch/contact*"
```
<details><summary>output</summary>

```
33
-rw-rw-r-- 1 hanzhou hanzhou 713K Oct 20 19:57 /work/scratch/contact_00.jpg
-rw-rw-r-- 1 hanzhou hanzhou 259K Oct 20 19:57 /work/scratch/contact_01.jpg

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'for f in /pipeline/.stages/*.json /pipeline/pipeline_report.json /pipeline/depth/moge2/depth_report.json; do echo ===$f===; cat $f; done'
```
<details><summary>output</summary>

```
===/pipeline/.stages/01_depth.json===
{
  "name": "01_depth",
  "outputs": [
    {
      "mtime_ns": 1791211350850777025,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000005/depth/moge2/raw.npy",
      "size": 3549560960
    },
    {
      "mtime_ns": 1791211351213777024,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000005/depth/moge2/intrinsics.pkl",
      "size": 106729
    },
    {
      "mtime_ns": 1791211354162777013,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000005/depth/moge2/depth_report.json",
      "size": 1815
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/mhr_wild_depth.py",
      "--video",
      "/home/hanzhou/Projects/v2d/run/ …[+24236 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz')
for k in ['T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 T=z[k]; print('\\n',k)
 for i in [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1002]:
  R=T[i,:3,:3]; t=T[i,:3,3]
  # rotation vector
  from scipy.spatial.transform import Rotation
  rv=Rotation.from_matrix(R).as_rotvec()
  print(i, 't',np.round(t,3),'rv',np.round(rv,3))
print('contact probs')
cp=z['contact_prob_right_left'];print(cp[[0,100,200,300,400,500,600,700,800,900,1002]])
for i in [0,400,800,1002]: print(i, z['human_keypoints_world'][i,[21,41,42,62]])
PY"
```
<details><summary>output</summary>

```

 T_world_mesh
0 t [-0.121  0.783  2.706] rv [-0.863  1.525  1.588]
50 t [-0.123  0.791  2.766] rv [-0.859  1.548  1.581]
100 t [-0.113  0.782  2.726] rv [-0.836  1.54   1.606]
150 t [-0.116  0.787  2.737] rv [-0.843  1.554  1.587]
200 t [-0.046  0.759  2.844] rv [-0.961  1.707  1.595]
250 t [0.041 0.765 2.986] rv [-1.508  1.916  1.296]
300 t [0.162 0.892 3.219] rv [ 1.996 -2.087 -0.641]
350 t [0.183 0.927 3.322] rv [ 1.916 -2.024 -0.652]
400 t [0.176 0.912 3.229] rv [ 1.957 -2.008 -0.58 ]
450 t [0.175 0.951 3.388] rv [ 1.931 -2.013 -0.608]
500 t [0.294 0.867 3.166] rv [ 2.46  -1.496 -0.792]
550 t [0.53  0.94  3.038] rv [-2.931  0.103  0.661]
600 t [0.58  0.98  3.074] rv [-2.952  0.014  0.554]
650 t [0.588 0.967 3.066] rv [-2.954  0.011  0.561]
700 t [0.589 0.986 3.094] rv [-2.967  0.027   …[+3572 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, pprint
for f in ['/pipeline/foundationpose/episode_000005.pkl','/pipeline/mhr_init/episode_000005.stats.pkl']:
 print('===',f)
 try:
  x=pickle.load(open(f,'rb')); print(type(x)); print(str(x)[:5000])
 except Exception as e: print(e)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analyze_proj.py <<'PY'
import numpy as np, cv2, trimesh, h5py, json
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; F=len(T)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=mesh.vertices[::20]
with h5py.File('/work/inputs/masks.h5','r') as f: om=f['object'][:]; pm=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4')

def hull(Tm):
    pc=(Tcw@Tm)[:3,:3]@V.T+(Tcw@Tm)[:3,3,None]
    uv=(K@pc); uv=(uv[:2]/uv[2]).T
    h=cv2.convexHull(uv.astype(np.float32)); m=np.zeros((1152,1536),np.uint8); cv2.fillPoly(m,[np.round(h).astype(np.int32)],1)
    return m.astype(bool),uv

def iou(a,b): return (a&b).sum()/max(1,(a|b).sum())
segs=[]
for i in range(F):
    m,uv=hull(T[i]); o=om[i]
    # visible target: object excluding person occlusion approx
    p=pm[i]; target=o&~p
    segs.append(iou(target,m))
segs=np.array(segs)
print('iou stats',np.nanmin(segs),np.median(segs),np.nanmean(segs),np.nanmax(segs))
print('worst')
for i in np.argsort(segs)[:50]: print(i,round(float(segs[i]),3),'obj',om[i].sum(),'pers',pm[i].sum(),'T',np.round(T[i,:3,3],3))
print('per 25',[(i,round(float(np.median(segs[i:i+25])),3)) for i in range(0,F,25)])
# overlay 1 fps
import os
os.makedirs('/work/scratch/ov',exist_ok=True)
for  …[+1013 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/analyze_proj.py", line 46
    Cw=np.eye(4); Cw[:3,:3]=Rwc:=Twc[:3,:3]@C[i,:3,:3]; Cw[:3,3]=T[i,:3,3]
                               ^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/Cw\\[:3,:3\\]=Rwc:=Twc\\[:3,:3\\]@C\\[i,:3,:3\\]/Rwc=Twc[:3,:3]@C[i,:3,:3]; Cw[:3,:3]=Rwc/' /work/scratch/analyze_proj.py
python /work/scratch/analyze_proj.py"
```
<details><summary>output</summary>

```
iou stats 0.7884058777691194 0.8674116447164205 0.871551346878848 0.9359074719167091
worst
543 0.788 obj 28329 pers 16695 T [0.519 0.919 2.932]
544 0.798 obj 28382 pers 16730 T [0.522 0.922 2.951]
542 0.805 obj 28807 pers 16547 T [0.52  0.918 2.941]
540 0.807 obj 28701 pers 16546 T [0.512 0.917 2.943]
548 0.811 obj 28689 pers 16295 T [0.521 0.925 2.958]
821 0.811 obj 39844 pers 30612 T [0.532 0.817 2.776]
556 0.811 obj 28675 pers 16791 T [0.52  0.922 2.955]
822 0.811 obj 40128 pers 30547 T [0.526 0.816 2.77 ]
254 0.814 obj 34801 pers 25861 T [0.054 0.764 2.997]
257 0.814 obj 34176 pers 25686 T [0.067 0.772 3.022]
826 0.814 obj 41008 pers 30447 T [0.518 0.802 2.753]
549 0.815 obj 28815 pers 16779 T [0.525 0.925 2.963]
248 0.815 obj 35707 pers 26463 T [0.036 0.762 2.966]
277 0.815 obj 29249  …[+2748 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image,ImageDraw
import glob,os
files=sorted(glob.glob('/work/scratch/ov/*.jpg'))
for ci in range(0,len(files),20):
 fs=files[ci:ci+20]; sh=Image.new('RGB',(480*4,360*5),'white');d=ImageDraw.Draw(sh)
 for j,f in enumerate(fs):
  im=Image.open(f).resize((480,360)); sh.paste(im,((j%4)*480,(j//4)*360));d.text(((j%4)*480+5,(j//4)*360+5),os.path.basename(f),fill='white',stroke_width=2,stroke_fill='black')
 sh.save(f'/work/scratch/ov_contact_{ci//20}.jpg',quality=92)
print(len(files))
PY"
```
<details><summary>output</summary>

```
34

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ls -lh /work/scratch | head -20; find /work/scratch -maxdepth 1 -name 'ov_contact*' -printf '%p\\n'"
```
<details><summary>output</summary>

```
total 2.7M
-rw-rw-r-- 1 hanzhou hanzhou  2.4K Oct 20 19:57 analyze_proj.py
-rw-rw-r-- 1 hanzhou hanzhou  713K Oct 20 19:57 contact_00.jpg
-rw-rw-r-- 1 hanzhou hanzhou  259K Oct 20 19:57 contact_01.jpg
drwxrwxr-x 2 hanzhou hanzhou  4.0K Oct 20 19:57 frames
drwxrwxr-x 2 hanzhou hanzhou  4.0K Oct 20 19:57 ov
-rw-rw-r-- 1 hanzhou hanzhou 1003K Oct 20 19:58 ov_contact_0.jpg
-rw-rw-r-- 1 hanzhou hanzhou  719K Oct 20 19:58 ov_contact_1.jpg
/work/scratch/ov_contact_0.jpg
/work/scratch/ov_contact_1.jpg

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/plot_stats.py <<'PY'
import numpy as np,h5py,cv2,json,matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];F=len(T);t=T[:,:3,3]
with h5py.File('/work/inputs/masks.h5','r') as f: om=f['object'][:]; pm=f['person'][:]
cent=[];box=[];area=[];pcent=[]
for i in range(F):
 ys,xs=np.where(om[i]); area.append(len(xs));cent.append([xs.mean(),ys.mean()]);box.append([xs.min(),ys.min(),xs.max(),ys.max()])
 ys,xs=np.where(pm[i]);pcent.append([xs.mean(),ys.mean()])
cent=np.array(cent);box=np.array(box);area=np.array(area);pcent=np.array(pcent)
# frame image difference on visible object? no
fig,axs=plt.subplots(5,1,figsize=(18,18),sharex=True)
axs[0].plot(t);axs[0].legend(['tx','ty','tz']);axs[0].grid()
axs[1].plot(cent);axs[1].legend(['cx','cy']);axs[1].grid(); axs[1].set_ylim(1152,0)
axs[2].plot(box);axs[2].legend(['x0','y0','x1','y1']);axs[2].grid(); axs[2].set_ylim(1152,0)
axs[3].plot(area);axs[3].grid();
axs[4].plot(pcent);axs[4].legend(['pcx','pcy']);axs[4].grid();axs[4].set_ylim(1152,0)
for ax in axs:
 for x in range(0,F,50): ax.axvline(x,color='k',alpha=.1)
fig.tight_layout();fig.savefig('/work/scratch/stats.png',dpi=100)
# print every 10
print('i t baseline area obj center pcenter')
for i in range(0,F,10): print(i,*np.round(t[i],3),area[i],*np.round(cent[i],1),*np.round(pcent[i],1))
PY
python /work/scratch/plot_stats.py > /work/scratch/stats.txt"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print(s); print('geom',s.geometry); 
for k,g in s.geometry.items():
 print(k,'visual',type(g.visual),g.visual)
 print('uv',g.visual.uv.shape if hasattr(g.visual,'uv') and g.visual.uv is not None else None)
 print('mat',g.visual.material)
 print('tex',getattr(g.visual.material,'image',None),getattr(g.visual.material,'baseColorTexture',None))
PY"
```
<details><summary>output</summary>

```
<trimesh.Scene(len(geometry)=1)>
geom OrderedDict([('bigblackbox', <trimesh.Trimesh(vertices.shape=(289270, 3), faces.shape=(96454, 3))>)])
bigblackbox visual <class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.texture.TextureVisuals object at 0x7a3d04bee450>
uv (289270, 2)
mat <trimesh.visual.material.PBRMaterial object at 0x7a3ce551ba50>
tex None <PIL.PngImagePlugin.PngImageFile image mode=RGB size=5007x4408 at 0x7A3CBB1FB290>

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_test.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py
import nvdiffrast.torch as dr
from PIL import Image
device='cuda'; torch.manual_seed(0)
W,H=1536,1152
K=np.array(json.load(open('/work/inputs/camera.json'))['K'],dtype=np.float32)
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'],dtype=np.float32); Tcw=np.linalg.inv(Twc).astype(np.float32)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(g.vertices,dtype=torch.float32,device=device)
Faces=torch.tensor(g.faces,dtype=torch.int32,device=device)
uv=torch.tensor(g.visual.uv,dtype=torch.float32,device=device)
tex=np.asarray(g.visual.material.baseColorTexture).astype(np.float32)/255.
tex=torch.tensor(tex,dtype=torch.float32,device=device).permute(2,0,1)[None]
ctx=dr.RasterizeCudaContext()
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as f: om=f['object'][:];pm=f['person'][:]

def get_frame(i):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

def render(Twm):
 Twc_t=torch.tensor(Twc,device=device); Tcwt=torch.tensor(Tcw,device=device);Tmt=torch.tensor(Twm,dtype=torch.float32,device=device)
 vt=V@Tmt[:3,:3].T+Tmt[:3,3]
 vc=vt@Tcwt[:3,:3].T+Tcwt[:3,3]
 zz=vc[:,2]
 u=K[0,0]*vc[:,0]/zz+K[0,2]; v=K[1,1]*vc[:,1]/zz+K[1,2]
 clip=torch.stack([2*u/W-1, …[+1888 chars]
```
<details><summary>output</summary>

```
FRAME 0
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 59, in <module>
    iou,mae,n,rgb,mask=score(i,Tw)
                       ^^^^^^^^^^^
  File "/work/scratch/render_test.py", line 39, in score
    im=get_frame(i).astype(np.float32)/255; rgb,mask,d=render(Tm)
                                                       ^^^^^^^^^^
  File "/work/scratch/render_test.py", line 31, in render
    iuv=dr.interpolate(uvh,rast,Faces)[0]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
 …[+417 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py'
s=open(p).read();s=s.replace(\"uvh=torch.cat([uv,torch.ones_like(uv[:,:1])],1).permute(1,0)[None]\",\"uvh=torch.stack([uv[:,0],uv[:,1],torch.ones_like(uv[:,0])],dim=-1)[None].contiguous()\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
FRAME 0
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 59, in <module>
    iou,mae,n,rgb,mask=score(i,Tw)
                       ^^^^^^^^^^^
  File "/work/scratch/render_test.py", line 39, in score
    im=get_frame(i).astype(np.float32)/255; rgb,mask,d=render(Tm)
                                                       ^^^^^^^^^^
  File "/work/scratch/render_test.py", line 34, in render
    rgb=torch.nn.functional.grid_sample(tex,grid,align_corners=False,mode='bilinear',padding_mode='border')[0].permute(1,2,0)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/functional.py", line 4910, in grid_sample
    return torch.grid_sampler(input, grid, mode …[+304 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read();s=s.replace(\"iuv=dr.interpolate(uvh,rast,Faces)[0]\\n mx=\", \"iuv=dr.interpolate(uvh,rast,Faces)[0]\\n print('iuv shape',iuv.shape,'rast',rast.shape)\\n mx=\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py 2>&1 | head"
```
<details><summary>output</summary>

```
FRAME 0
iuv shape torch.Size([1, 1152, 1536, 3]) rast torch.Size([1, 1152, 1536, 4])
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 60, in <module>
    iou,mae,n,rgb,mask=score(i,Tw)
                       ^^^^^^^^^^^
  File "/work/scratch/render_test.py", line 40, in score
    im=get_frame(i).astype(np.float32)/255; rgb,mask,d=render(Tm)
                                                       ^^^^^^^^^^
  File "/work/scratch/render_test.py", line 35, in render

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read();s=s.replace(\"iuv=dr.interpolate(uvh,rast,Faces)[0]\\n print('iuv shape',iuv.shape,'rast',rast.shape)\", \"iuv=dr.interpolate(uvh,rast,Faces)[0][0]\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
FRAME 0
none 0.0 9.0 0
fx 0.0 9.0 0
fy 0.0 9.0 0
fz 0.0 9.0 0
rz90 0.0 9.0 0
rz270 0.0 9.0 0
FRAME 50
none 0.0 9.0 0
fx 0.0 9.0 0
fy 0.0 9.0 0
fz 0.0 9.0 0
rz90 0.0 9.0 0
rz270 0.0 9.0 0
FRAME 100
none 0.0 9.0 0
fx 0.0 9.0 0
fy 0.0 9.0 0
fz 0.0 9.0 0
rz90 0.0 9.0 0
rz270 0.0 9.0 0
FRAME 700
none 0.0 9.0 0
fx 0.0 9.0 0
fy 0.0 9.0 0
fz 0.0 9.0 0
rz90 0.0 9.0 0
rz270 0.0 9.0 0
FRAME 950
none 0.0 9.0 0
fx 0.0 9.0 0
fy 0.0 9.0 0
fz 0.0 9.0 0
rz90 0.0 9.0 0
rz270 0.0 9.0 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/debug_raster.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"if __name__\")[0])
# modify render debug
Tmt=torch.tensor(T[0],dtype=torch.float32,device=device); vt=V@Tmt[:3,:3].T+Tmt[:3,3]; Tcwt=torch.tensor(Tcw,device=device);vc=vt@Tcwt[:3,:3].T+Tcwt[:3,3];zz=vc[:,2]
u=K[0,0]*vc[:,0]/zz+K[0,2];v=K[1,1]*vc[:,1]/zz+K[1,2]
clip=torch.stack([2*u/W-1,1-2*v/H,zz/10,torch.ones_like(zz)],1)[None]
print(clip.min(1)[0],clip.max(1)[0],clip.shape,Faces.shape,clip.dtype,Faces.dtype)
rast,hard=dr.rasterize(ctx,clip,Faces,resolution=[H,W]);print('rast',rast.shape,rast.min(),rast.max(),hard.shape,(rast[...,3]>0).sum(),(hard>0).sum())
PY
python /work/scratch/debug_raster.py"
```
<details><summary>output</summary>

```
tensor([[-0.2526, -0.6460,  0.2195,  1.0000]], device='cuda:0') tensor([[-0.0013, -0.1000,  0.2824,  1.0000]], device='cuda:0') torch.Size([1, 289270, 4]) torch.Size([96454, 3]) torch.float32 torch.int32
rast torch.Size([1, 1152, 1536, 4]) tensor(0., device='cuda:0') tensor(96454., device='cuda:0') torch.Size([1, 1152, 1536, 4]) tensor(50625, device='cuda:0') tensor(95277, device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/scratch/render_test.py | sed -n '20,65p'; python - <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"if __name__\")[0])
r,m,d=render(T[0]);print('masks',m.sum(),om[0].sum(),pm[0].sum(),(om[0]&~pm[0]).sum(),(m&(om[0]&~pm[0])).sum());print(r.shape,m.shape)
PY"
```
<details><summary>output</summary>

```
masks 50625 48579 28056 48579 0
(1152, 1536, 3) (1152, 1536)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 o=f['object'][0];p=f['person'][0]
 ys,xs=np.where(o);print('o bbox',xs.min(),xs.max(),ys.min(),ys.max())
 ys,xs=np.where(p);print('p bbox',xs.min(),xs.max(),ys.min(),ys.max())
 print('overlap',(o&p).sum(),'p-o',(p&~o).sum())
 # print row counts / actual types
PY"
```
<details><summary>output</summary>

```
o bbox 576 759 638 942
p bbox 775 892 402 815
overlap 0 p-o 28056

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"if __name__\")[0])
r,m,d=render(T[0]);ys,xs=np.where(m);print(m.sum(),xs.min(),xs.max(),ys.min(),ys.max());ys,xs=np.where(om[0]);o=np.zeros_like(m);o[ys,xs]=1;print('intersection',(m&o).sum(),'union',(m|o).sum());print('m first',np.argwhere(m)[:5], 'o first',np.argwhere(o)[:5])
PY"
```
<details><summary>output</summary>

```
50625 574 766 204 517
intersection 0 union 99204
m first [[204 686]
 [204 687]
 [204 688]
 [205 683]
 [205 684]] o first [[638 661]
 [638 662]
 [638 663]
 [638 664]
 [638 665]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read().replace('1-2*v/H','2*v/H-1');open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
FRAME 0
none 0.9477 0.1631 48271
fx 0.8311 0.1435 45002
fy 0.9238 0.1421 47817
fz 0.8565 0.1495 45757
rz90 0.8847 0.1673 48096
rz270 0.8932 0.1541 47280
FRAME 50
none 0.9406 0.1588 46739
fx 0.8419 0.1424 44034
fy 0.8976 0.1381 45788
fz 0.8804 0.1468 45110
rz90 0.8934 0.1649 46907
rz270 0.8961 0.1529 45970
FRAME 100
none 0.9304 0.1579 47339
fx 0.8524 0.1408 45131
fy 0.8944 0.1354 46530
fz 0.8844 0.1436 46030
rz90 0.8834 0.161 47378
rz270 0.8806 0.1481 46263
FRAME 700
none 0.9339 0.1643 27388
fx 0.8112 0.1569 25104
fy 0.9253 0.1489 27311
fz 0.8313 0.1426 25631
rz90 0.749 0.1593 23693
rz270 0.7251 0.1473 23023
FRAME 950
none 0.9075 0.1411 58183
fx 0.8563 0.1752 58207
fy 0.8875 0.1281 57932
fz 0.8681 0.1762 58207
rz90 0.8522 0.1528 53843
rz270 0.8307 0.1464 53194

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image
import numpy as np,trimesh
im=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().visual.material.baseColorTexture
im.resize((1000,881)).save('/work/scratch/texture.jpg')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,50,100,200,300,400,500,600,700,800,900,950]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();cv2.imwrite(f'/work/scratch/crop_{i}.jpg',im[500:1050,450:1000])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read()
# add flip y in grid
s=s.replace(\"grid=torch.stack([mx*2-1,my*2-1],-1)[None]\",\"grid=torch.stack([mx*2-1,1-my*2],-1)[None]\")
open('/work/scratch/render_test_flip.py','w').write(s)
PY
python /work/scratch/render_test_flip.py"
```
<details><summary>output</summary>

```
FRAME 0
none 0.9477 0.2272 48271
fx 0.8311 0.2488 45002
fy 0.9238 0.1926 47817
fz 0.8565 0.1974 45757
rz90 0.8847 0.184 48096
rz270 0.8932 0.2237 47280
FRAME 50
none 0.9406 0.2252 46739
fx 0.8419 0.2441 44034
fy 0.8976 0.1894 45788
fz 0.8804 0.1961 45110
rz90 0.8934 0.1821 46907
rz270 0.8961 0.2223 45970
FRAME 100
none 0.9304 0.2335 47339
fx 0.8524 0.249 45131
fy 0.8944 0.1913 46530
fz 0.8844 0.2009 46030
rz90 0.8834 0.1866 47378
rz270 0.8806 0.2267 46263
FRAME 700
none 0.9339 0.1926 27388
fx 0.8112 0.2237 25104
fy 0.9253 0.2281 27311
fz 0.8313 0.1868 25631
rz90 0.749 0.1966 23693
rz270 0.7251 0.1918 23023
FRAME 950
none 0.9075 0.2357 58183
fx 0.8563 0.205 58207
fy 0.8875 0.2112 57932
fz 0.8681 0.1995 58207
rz90 0.8522 0.267 53843
rz270 0.8307 0.2247 53194

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];t=T[:,:3,3]
# relative to med 0:100
ref=np.median(t[:100],axis=0)
print('ref',ref)
for i in range(80,221): print(i,np.round(t[i]-ref,3))
PY"
```
<details><summary>output</summary>

```
[ 0.005 -0.006 -0.011]
82 [ 0.007 -0.008 -0.019]
83 [ 0.008 -0.011 -0.031]
84 [ 0.005 -0.006 -0.006]
85 [ 0.008 -0.005 -0.019]
86 [ 0.009 -0.005 -0.016]
87 [ 0.005 -0.002 -0.01 ]
88 [ 0.007 -0.002 -0.012]
89 [ 0.007 -0.002 -0.016]
90 [ 0.008 -0.004 -0.022]
91 [ 0.008 -0.004 -0.005]
92 [ 0.008 -0.006 -0.015]
93 [ 0.008 -0.003 -0.015]
94 [ 0.006 -0.002 -0.008]
95 [ 0.006 -0.002 -0.017]
96 [ 0.005 -0.004 -0.035]
97 [ 0.006 -0.003 -0.029]
98 [ 0.007 -0.003 -0.027]
99 [ 0.007 -0.004 -0.02 ]
100 [ 0.007 -0.003 -0.017]
101 [ 0.007 -0.005 -0.024]
102 [ 0.006 -0.006 -0.038]
103 [ 0.005 -0.004 -0.032]
104 [ 0.008 -0.005 -0.036]
105 [ 0.006 -0.003 -0.018]
106 [ 0.007 -0.006 -0.031]
107 [ 0.005 -0.    -0.017]
108 [ 0.006  0.002 -0.007]
109 [ 0.005 -0.001 -0.024]
110 [ 0.007 -0.002 -0.029]
111 [ 0.008  …[+2949 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];t=T[:,:3,3];ref=np.median(t[960:],axis=0);print(ref)
for i in range(840,1003,4): print(i,np.round(t[i]-ref,3), round(np.linalg.norm(t[i]-ref),3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/photo_analysis.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"if __name__\")[0])
from scipy.spatial.transform import Rotation
import numpy as np, cv2, torch

def render2(Twm,uvflip=False):
 Tmt=torch.tensor(Twm,dtype=torch.float32,device=device); vt=V@Tmt[:3,:3].T+Tmt[:3,3]; Tcwt=torch.tensor(Tcw,device=device);vc=vt@Tcwt[:3,:3].T+Tcwt[:3,3];zz=vc[:,2]
 u=K[0,0]*vc[:,0]/zz+K[0,2];v=K[1,1]*vc[:,1]/zz+K[1,2]
 clip=torch.stack([2*u/W-1,2*v/H-1,zz/10,torch.ones_like(zz)],1)[None]
 rast,hard=dr.rasterize(ctx,clip,Faces,resolution=[H,W])
 uvh=torch.stack([uv[:,0],uv[:,1],torch.ones_like(uv[:,0])],dim=-1)[None].contiguous(); iuv=dr.interpolate(uvh,rast,Faces)[0][0]
 mx=iuv[...,0]/iuv[...,2]; my=iuv[...,1]/iuv[...,2]
 if uvflip: my=1-my
 grid=torch.stack([mx*2-1,my*2-1],-1)[None]
 rgb=torch.nn.functional.grid_sample(tex,grid,align_corners=False,mode='bilinear',padding_mode='border')[0].permute(1,2,0)
 mask=(rast[0,...,3]>0)
 return rgb.detach().cpu().numpy(),mask.detach().cpu().numpy()

def metrics(im,rgb,mask,valid):
 e=mask&valid
 if e.sum()<100:return [0]*6
 o=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32);r=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
 # gradient
 ogx=cv2.Sobel(o,cv2.CV_32F,1,0,ksize=3);ogy=cv2.Sobel(o,cv2.CV_32F,0,1,ksize=3)
 rgx=cv2.Sobel(r,cv2.CV_32F,1,0,ksize=3);rgy=cv2.Sobel(r,cv2.CV_32F,0,1,ksize=3)
 def corr(a,b,e):
  aa=a[e];bb=b[e];return np.corrcoef(aa,bb)[0,1]
 # normalized lo …[+1128 chars]
```
<details><summary>output</summary>

```
FRAME 0
none -0.101 0.009 -0.020 0.010 23.035 21.940
none_uvf 0.043 -0.086 -0.028 0.214 22.896 21.801
fx -0.237 0.008 0.029 0.030 22.957 21.848
fy -0.083 -0.129 0.008 0.181 23.035 21.940
fz 0.333 0.024 0.059 0.103 22.937 21.832
rz90 0.289 -0.028 -0.018 0.122 22.870 21.778
rz270 0.264 0.002 0.007 0.076 23.260 22.169
FRAME 30
none -0.085 -0.014 -0.024 0.054 22.866 21.874
none_uvf 0.030 -0.177 -0.019 0.309 22.728 21.737
fx -0.234 0.004 -0.028 0.026 22.946 21.926
fy -0.078 -0.133 -0.021 0.210 23.009 22.019
fz 0.332 0.014 0.045 0.090 22.890 21.872
rz90 0.303 0.025 -0.004 0.123 22.672 21.683
rz270 0.272 0.009 0.023 0.076 23.103 22.122
FRAME 60
none -0.082 -0.009 -0.023 0.027 22.110 21.191
none_uvf -0.017 0.009 -0.010 0.156 21.970 21.052
fx -0.258 -0.030 -0.006 0.030 21.852 20.930
fy -0.176 0.002 …[+1131 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_candidates.py <<'PY'
exec(open('/work/scratch/photo_analysis.py').read().split(\"flips=\")[0])
from scipy.spatial.transform import Rotation
import cv2, numpy as np
for i in [0,950]:
 im=get_frame(i);valid=om[i]&~pm[i]
 for name,Rf in [('none',np.eye(3)),('fz',Rotation.from_euler('z',180,degrees=True).as_matrix()),('fy',Rotation.from_euler('y',180,degrees=True).as_matrix()),('fx',Rotation.from_euler('x',180,degrees=True).as_matrix())]:
  for uf in [False,True]:
   Tw=np.array(T[i]);Tw[:3,:3]=Tw[:3,:3]@Rf
   rgb,mask=render2(Tw,uf)
   out=im.copy();out[mask]=(0.5*out[mask]+0.5*rgb[mask]*255).astype(np.uint8)
   cv2.imwrite(f'/work/scratch/renders/c_{i}_{name}_{int(uf)}.jpg',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
PY
python /work/scratch/render_candidates.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image,ImageDraw
import os
for i in [0,950]:
 fs=[f'/work/scratch/renders/c_{i}_{n}_{u}.jpg' for n in ['none','fx','fy','fz'] for u in [0,1]]
 sh=Image.new('RGB',(550*4,550*2))
 d=ImageDraw.Draw(sh)
 for j,f in enumerate(fs):
  im=Image.open(f).crop((450,500,1000,1050));sh.paste(im,((j%4)*550,(j//4)*550));d.text(((j%4)*550+5,(j//4)*550+5),os.path.basename(f),fill='red',stroke_width=2,stroke_fill='white')
 sh.save(f'/work/scratch/cand_{i}.jpg',quality=95)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"if __name__\")[0])
from scipy.spatial.transform import Rotation
for name,Rf in [('none',np.eye(3)),('fx',Rotation.from_euler('x',180,degrees=True).as_matrix()),('fy',Rotation.from_euler('y',180,degrees=True).as_matrix()),('fz',Rotation.from_euler('z',180,degrees=True).as_matrix())]:
 Tw=np.array(T[0]);Tw[:3,:3]=Tw[:3,:3]@Rf
 rgb,m,d=render(Tw);ys,xs=np.where(m);print(name,m.sum(),xs.min(),xs.max(),ys.min(),ys.max(),np.mean(np.argwhere(m),0))
PY"
```
<details><summary>output</summary>

```
none 50625 574 766 634 947 [781.68296296 669.56134321]
fx 50570 563 750 612 938 [773.7195175  657.28617758]
fy 50997 576 766 615 938 [775.23403337 669.13891013]
fz 50600 562 758 633 948 [780.83405138 657.80179842]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/search_sym.py <<'PY'
exec(open('/work/scratch/photo_analysis.py').read().split(\"flips=\")[0])
import itertools, numpy as np, cv2
from scipy.spatial.transform import Rotation
# all signed permutation rotation matrices det +1
Rs=[]
for perm in itertools.permutations(range(3)):
 for signs in itertools.product([-1,1],repeat=3):
  R=np.zeros((3,3))
  for j,(i,s) in enumerate(zip(perm,signs)): R[i,j]=s
  if np.linalg.det(R)>0: Rs.append(R)
print('num',len(Rs))
# local axis rotations full group but unique
def corrmetrics(im,rgb,mask,valid):
 e=mask&valid
 if e.sum()<100:return 0,0,0,0,e.sum()
 o=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32);r=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
 ogx=cv2.Sobel(o,cv2.CV_32F,1,0,ksize=3);ogy=cv2.Sobel(o,cv2.CV_32F,0,1,ksize=3);rgx=cv2.Sobel(r,cv2.CV_32F,1,0,ksize=3);rgy=cv2.Sobel(r,cv2.CV_32F,0,1,ksize=3)
 def c(a,b):
  aa=a[e];bb=b[e]
  return np.corrcoef(aa,bb)[0,1] if aa.std()>1e-6 and bb.std()>1e-6 else 0
 return c(o,r),c(ogx,rgx),c(ogy,rgy),np.corrcoef(np.hypot(ogx,ogy)[e],np.hypot(rgx,rgy)[e])[0,1],e.sum()
frames=[0,30,60,90,940,960,980,1000]
ims={i:get_frame(i) for i in frames};valids={i:om[i]&~pm[i] for i in frames}
allm=[]
for ri,Rf in enumerate(Rs):
 rows=[]
 for i in frames:
  Tw=np.array(T[i]);Tw[:3,:3]=Tw[:3,:3]@Rf
  rgb,mask=render2(Tw,False)
  m=corrmetrics(ims[i],rgb,mask,valids[i]);
  # adjusted mask alignment IoU after small t shift? no
  inter=(mask&valids[i]).sum(); …[+575 chars]
```
<details><summary>output</summary>

```
num 24
0 [[-1, 0, 0], [0, -1, 0], [0, 0, 1]] corr [-0.079 -0.068 -0.077 -0.067 -0.267 -0.28  -0.274 -0.262] grad [-0.01   0.001  0.06 ] iou [45757. 45446. 45837. 46293. 57856. 58187. 58517. 58815.] n [0 0 0 0 0 0 0 0]
1 [[-1, 0, 0], [0, 1, 0], [0, 0, -1]] corr [0.061 0.076 0.112 0.13  0.134 0.122 0.124 0.118] grad [-0.017  0.006  0.095] iou [47817. 46791. 45667. 46619. 57545. 58048. 58244. 58414.] n [0 0 0 0 0 0 0 0]
2 [[1, 0, 0], [0, -1, 0], [0, 0, -1]] corr [ 0.082  0.085  0.063  0.049 -0.022 -0.04  -0.046 -0.054] grad [-0.004 -0.     0.052] iou [45002. 44499. 44695. 45376. 57856. 58187. 58516. 58815.] n [0 0 0 0 0 0 0 0]
3 [[1, 0, 0], [0, 1, 0], [0, 0, 1]] corr [-0.101 -0.085 -0.082 -0.113 -0.028 -0.031 -0.018 -0.004] grad [-0.014 -0.001  0.069] iou [48271. 47619. 46671. 47528. 57774. 5 …[+5604 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,trimesh
from trimesh.visual import color
print(inspect.getsource(color.uv_to_color))
PY"
```
<details><summary>output</summary>

```
def uv_to_color(uv, image) -> NDArray[np.uint8]:
    """
    Get the color in a texture image.

    Parameters
    -------------
    uv : (n, 2) float
      UV coordinates on texture image
    image : PIL.Image
      Texture image

    Returns
    ----------
    colors : (n, 4) uint4
      RGBA color at each of the UV coordinates
    """
    if image is None or uv is None:
        return None

    # UV coordinates should be (n, 2) float
    uv = np.asanyarray(uv, dtype=np.float64)

    # get texture image pixel positions of UV coordinates
    x = (uv[:, 0] * (image.width - 1)) % image.width
    y = ((1 - uv[:, 1]) * (image.height - 1)) % image.height

    # access colors from pixel locations
    # make sure image is RGBA before getting values
    colors = np.asanyarray(image.convert("RGBA" …[+274 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/photo_sample.py <<'PY'
exec(open('/work/scratch/photo_analysis.py').read().split(\"flips=\")[0])
from scipy.spatial.transform import Rotation
import numpy as np,cv2
flips={'none':np.eye(3),'fx':Rotation.from_euler('x',180,degrees=True).as_matrix(),'fy':Rotation.from_euler('y',180,degrees=True).as_matrix(),'fz':Rotation.from_euler('z',180,degrees=True).as_matrix(),'rz90':Rotation.from_euler('z',90,degrees=True).as_matrix(),'rz270':Rotation.from_euler('z',270,degrees=True).as_matrix()}
def score(im,rgb,mask,valid):
 e=mask&valid
 if e.sum()<100:return -2,-2
 o=im[e].astype(np.float32);p=rgb[e].astype(np.float32)
 # fit per channel affine, calculate normalized RMSE and corr
 cs=[];nrms=[]
 for c in range(3):
  a=np.polyfit(p[:,c],o[:,c],1);res=o[:,c]-(p[:,c]*a[0]+a[1]);nrms.append(np.sqrt((res**2).mean())/(o[:,c].std()+1e-4))
  cs.append(np.corrcoef(o[:,c],p[:,c])[0,1] if o[:,c].std()>1e-4 and p[:,c].std()>1e-4 else 0)
 og=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32);rg=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
 gx1=cv2.Sobel(og,cv2.CV_32F,1,0);gy1=cv2.Sobel(og,cv2.CV_32F,0,1);gx2=cv2.Sobel(rg,cv2.CV_32F,1,0);gy2=cv2.Sobel(rg,cv2.CV_32F,0,1)
 g=np.mean([np.corrcoef(gx1[e],gx2[e])[0,1],np.corrcoef(gy1[e],gy2[e])[0,1]]) if gx1[e].std()>1e-4 and gx2[e].std()>1e-4 else 0
 return np.mean(cs),g
wins={k:0 for k in flips}; tot={k:[] for k in flips}
for i in list(range(0,101,10))+list(range(125,1000,25)):
 im=get_frame(i);vali …[+451 chars]
```
<details><summary>output</summary>

```
0 fz none:0.04/-0.06 fx:-0.23/0.02 fy:-0.09/-0.06 fz:0.32/0.04 rz90:0.27/-0.02 rz270:0.26/0.00
10 fz none:0.02/-0.10 fx:-0.23/-0.00 fy:-0.08/-0.08 fz:0.31/0.02 rz90:0.28/-0.01 rz270:0.26/0.01
20 fz none:0.03/-0.11 fx:-0.23/-0.00 fy:-0.07/-0.09 fz:0.31/0.02 rz90:0.29/-0.00 rz270:0.27/0.01
30 fz none:0.02/-0.10 fx:-0.23/-0.01 fy:-0.08/-0.08 fz:0.32/0.03 rz90:0.29/0.01 rz270:0.27/0.02
40 fz none:0.02/-0.05 fx:-0.22/0.01 fy:-0.09/-0.03 fz:0.31/0.03 rz90:0.29/-0.00 rz270:0.26/0.01
50 fz none:0.01/-0.06 fx:-0.22/-0.01 fy:-0.11/0.01 fz:0.31/0.03 rz90:0.27/-0.01 rz270:0.25/-0.02
60 fz none:-0.02/-0.00 fx:-0.25/-0.02 fy:-0.17/-0.01 fz:0.25/-0.02 rz90:0.23/-0.03 rz270:0.27/-0.04
70 rz270 none:-0.06/0.01 fx:-0.28/-0.01 fy:-0.20/-0.04 fz:0.23/-0.02 rz90:0.22/-0.04 rz270:0.30/-0.02
80 rz270 none:-0.10/ …[+4277 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];
for i in [100,150,200,250,300,350,400,450,500,550,600,700,800,850,900,950,1002]:
 rv=R.from_matrix(T[0,:3,:3].T@T[i,:3,:3]).as_rotvec();print(i,np.round(rv,3),np.degrees(np.linalg.norm(rv)))
# axis angle change 0 to 1002 and components projected on initial world up axis? world y down, object local z? initial local z world vector
for i in [0,300,600,900,1002]: print(i,'local axes world',np.round(T[i,:3,:3],2))
PY"
```
<details><summary>output</summary>

```
100 [ 0.003 -0.008  0.027] 1.645126469085079
150 [0.017 0.005 0.022] 1.6163305375873762
200 [0.012 0.171 0.081] 10.886387999357215
250 [ 0.019  0.658 -0.067] 37.898931358614966
300 [ 0.049  1.457 -0.055] 83.56630139027317
350 [0.006 1.507 0.027] 86.35002783390284
400 [ 0.02   1.548 -0.017] 88.6854562504344
450 [0.014 1.536 0.01 ] 87.98422132021219
500 [-0.409  1.458 -0.384] 89.49048213260453
550 [-1.096  1.211 -1.271] 118.57520986914506
600 [-1.113  1.242 -1.362] 123.36266776042181
700 [-1.107  1.263 -1.357] 123.70212145577504
800 [-1.008  1.157 -1.437] 120.47992439193379
850 [-0.416  0.435 -1.615] 98.74414825871487
900 [ 0.021 -0.028 -1.625] 93.14935325532011
950 [ 0.016 -0.057 -1.58 ] 90.57606048272883
1002 [ 0.018 -0.06  -1.58 ] 90.61227160747802
0 local axes world [[-0.49 -0.87  0.03]
 …[+384 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/visual_fz.py <<'PY'
exec(open('/work/scratch/photo_analysis.py').read().split(\"flips=\")[0])
from scipy.spatial.transform import Rotation
import cv2,numpy as np
frames=[0,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
for i in frames:
 im=get_frame(i);out=[]
 for name,Rf in [('I',np.eye(3)),('Z180',Rotation.from_euler('z',180,degrees=True).as_matrix())]:
  Tw=np.array(T[i]);Tw[:3,:3]=Tw[:3,:3]@Rf;rgb,mask=render2(Tw,True)
  o=im.copy();o[mask]=(0.55*o[mask]+0.45*rgb[mask]*255).clip(0,255).astype(np.uint8)
  # crop around object based union mask/object
  ys,xs=np.where(mask);x0=max(0,xs.min()-30);x1=min(W,xs.max()+30);y0=max(0,ys.min()-30);y1=min(H,ys.max()+30)
  cr=o[y0:y1,x0:x1]; cr=cv2.resize(cr,(300,400));cv2.putText(cr,name,(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);out.append(cr)
 cv2.imwrite(f'/work/scratch/renders/fz_{i}.jpg',cv2.cvtColor(np.hstack(out),cv2.COLOR_RGB2BGR))
PY
python /work/scratch/visual_fz.py
python - <<'PY'
from PIL import Image,ImageDraw
import glob,os
fs=sorted(glob.glob('/work/scratch/renders/fz_*.jpg'),key=lambda x:int(x.split('_')[-1].split('.')[0]))
sh=Image.new('RGB',(600*4,400*5))
for j,f in enumerate(fs):
 im=Image.open(f);sh.paste(im,((j%4)*600,(j//4)*400));d=ImageDraw.Draw(sh);d.text(((j%4)*600+5,(j//4)*400+5),os.path.basename(f),fill='red',stroke_width=2,stroke_fill='white')
sh.save('/work/scratch/fz_contact.jpg',quality=95)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/edge_score.py <<'PY'
exec(open('/work/scratch/photo_analysis.py').read().split(\"flips=\")[0])
from scipy.spatial.transform import Rotation
import numpy as np,cv2
from scipy.ndimage import distance_transform_edt
from trimesh.geometry import face_normals
nV=torch.tensor(g.vertex_normals,dtype=torch.float32,device=device)
def render_geom(Twm):
 Tmt=torch.tensor(Twm,dtype=torch.float32,device=device); vt=V@Tmt[:3,:3].T+Tmt[:3,3]; Tcwt=torch.tensor(Tcw,device=device);vc=vt@Tcwt[:3,:3].T+Tcwt[:3,3];zz=vc[:,2]
 u=K[0,0]*vc[:,0]/zz+K[0,2];v=K[1,1]*vc[:,1]/zz+K[1,2]
 clip=torch.stack([2*u/W-1,2*v/H-1,zz/10,torch.ones_like(zz)],1)[None]
 rast,hard=dr.rasterize(ctx,clip,Faces,resolution=[H,W]); depth=rast[0,...,2]
 # normals camera
 vn=nV@Tmt[:3,:3].T; vn=vn@Tcwt[:3,:3].T
 nv=dr.interpolate(vn[None].contiguous(),rast,Faces)[0][0];nv=torch.nn.functional.normalize(nv,dim=-1)
 mask=rast[0,...,3]>0
 return depth.cpu().numpy(),nv.cpu().numpy(),mask.cpu().numpy()

def shift(mask,dx,dy):
 M=np.float32([[1,0,dx],[0,1,dy]]);return cv2.warpAffine(mask.astype(np.uint8),M,(W,H),flags=cv2.INTER_NEAREST)>0
def edges(depth,norm,mask):
 dep=cv2.normalize(np.nan_to_num(depth),[None],0,1,cv2.NORM_MINMAX).astype(np.float32)
 # depth edge on local changes
 gx=cv2.Sobel(dep,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(dep,cv2.CV_32F,0,1,ksize=3);e=np.hypot(gx,gy)
 # normal edge
 nx,ny,nz=[norm[...,i] for i in range(3)];eg=np.hypot(cv2.Sobel(nx,cv2.CV_32F,1,0),cv2.Sobel(nx,cv2. …[+1866 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/edge_score.py", line 5, in <module>
    from trimesh.geometry import face_normals
ImportError: cannot import name 'face_normals' from 'trimesh.geometry' (/opt/conda/lib/python3.11/site-packages/trimesh/geometry.py)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i '/from trimesh.geometry/d' /work/scratch/edge_score.py
python /work/scratch/edge_score.py"
```
<details><summary>output</summary>

```
FRAME 0
Traceback (most recent call last):
  File "/work/scratch/edge_score.py", line 57, in <module>
    sc=best_shift_edge(im,mt,dep,norm,mask);print(name,*[round(float(x),3) for x in sc])
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/edge_score.py", line 40, in best_shift_edge
    pe=edges(dep,norm,m)
       ^^^^^^^^^^^^^^^^^
  File "/work/scratch/edge_score.py", line 20, in edges
    dep=cv2.normalize(np.nan_to_num(depth),[None],0,1,cv2.NORM_MINMAX).astype(np.float32)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'normalize'
> Overload resolution failed:
>  - dst is not a numpy array, neither a scalar
>  - Expected Ptr<cv::UMat> for argument 'dst'


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/cv2.normalize(np.nan_to_num(depth),\\[None\\],0,1,cv2.NORM_MINMAX)/cv2.normalize(np.nan_to_num(depth),None,0,1,cv2.NORM_MINMAX)/' /work/scratch/edge_score.py
python /work/scratch/edge_score.py"
```
<details><summary>output</summary>

```
FRAME 0
none 0.948 11.718 1.425 6976.0 828.0 -0.863 1.354
fx 0.942 16.764 1.303 7632.0 828.0 11.412 9.317
fy 0.941 9.197 2.031 5664.0 828.0 -0.441 7.803
fz 0.948 22.8 1.527 7756.0 828.0 10.896 2.203
FRAME 30
none 0.957 9.481 0.696 6728.0 963.0 -1.58 0.517
fx 0.948 19.762 1.898 7741.0 963.0 10.442 8.293
fy 0.947 8.38 1.047 5841.0 963.0 -1.033 7.092
fz 0.952 22.616 3.015 7760.0 963.0 9.846 3.18
FRAME 60
none 0.961 9.095 0.725 7012.0 1071.0 -4.233 3.881
fx 0.951 19.519 1.95 7397.0 1071.0 5.726 9.7
fy 0.952 6.991 1.244 5678.0 1071.0 -3.813 10.43
fz 0.959 21.491 2.969 7945.0 1071.0 7.168 4.563
FRAME 90
none 0.955 11.656 0.956 7352.0 963.0 -3.965 3.712
fx 0.948 20.432 1.363 7481.0 963.0 8.237 9.61
fy 0.946 9.778 1.62 5853.0 963.0 -3.561 8.039
fz 0.954 22.964 1.784 8094.0 963.0 7.685 4.545
FRAME  …[+619 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/photo_analysis.py').read().split(\"flips=\")[0])
from scipy.spatial.transform import Rotation
import cv2,numpy as np
i=0;im=get_frame(i)
for name,Rf in [('none',np.eye(3)),('fz',Rotation.from_euler('z',180,degrees=True).as_matrix())]:
 Tw=np.array(T[i]);Tw[:3,:3]=Tw[:3,:3]@Rf;rgb,mask=render2(Tw,True)
 # shift fz by centroid to target
 ys,xs=np.where(mask);yt,xt=np.where(om[i]);dx=xt.mean()-xs.mean();dy=yt.mean()-ys.mean();M=np.float32([[1,0,dx],[0,1,dy]]);rgb=cv2.warpAffine(rgb,M,(W,H));mask=cv2.warpAffine(mask.astype(np.uint8),M,(W,H))>0
 # make side by side object crop + blend
 ys,xs=np.where(om[i]);x0=max(0,xs.min()-20);x1=xs.max()+20;y0=max(0,ys.min()-20);y1=ys.max()+20
 a=im[y0:y1,x0:x1];p=(rgb*255).clip(0,255).astype(np.uint8)[y0:y1,x0:x1];m=mask[y0:y1,x0:x1];p[~m]=im[y0:y1,x0:x1][~m]
 # normalize predicted global brightness to obs mean via green channel
 scale=np.median(a[m])/(np.median(p[m])+1e-3);pn=(p.astype(float)*scale).clip(0,255).astype(np.uint8)
 bl=a.copy();bl[m]=(0.5*a[m]+0.5*pn[m]).astype(np.uint8)
 out=np.hstack([a,pn,bl]);cv2.imwrite(f'/work/scratch/renders/correct_{name}.jpg',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
 print(name,'shift',dx,dy,'scale',scale)
PY"
```
<details><summary>output</summary>

```
none shift -0.8630785276064898 1.3539048194121506 scale 0.2499965278260024
fz shift 10.896466263297725 2.2028163989758696 scale 0.27272314055847635

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image
im=Image.open('/work/scratch/crop_0.jpg');im.resize((1100,1100)).save('/work/scratch/crop0_big.jpg')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# generate correct fz/identity at frame 950
exec(open('/work/scratch/photo_analysis.py').read().split(\"flips=\")[0])
from scipy.spatial.transform import Rotation
import cv2,numpy as np
i=950;im=get_frame(i)
for name,Rf in [('none',np.eye(3)),('fz',Rotation.from_euler('z',180,degrees=True).as_matrix())]:
 Tw=np.array(T[i]);Tw[:3,:3]=Tw[:3,:3]@Rf;rgb,mask=render2(Tw,True)
 ys,xs=np.where(mask);yt,xt=np.where(om[i]);dx=xt.mean()-xs.mean();dy=yt.mean()-ys.mean();M=np.float32([[1,0,dx],[0,1,dy]]);rgb=cv2.warpAffine(rgb,M,(W,H));mask=cv2.warpAffine(mask.astype(np.uint8),M,(W,H))>0
 ys,xs=np.where(om[i]);x0=max(0,xs.min()-20);x1=xs.max()+20;y0=max(0,ys.min()-20);y1=ys.max()+20
 a=im[y0:y1,x0:x1];p=(rgb*255).clip(0,255).astype(np.uint8)[y0:y1,x0:x1];m=mask[y0:y1,x0:x1];p[~m]=im[y0:y1,x0:x1][~m]
 scale=np.median(a[m])/(np.median(p[m])+1e-3);pn=(p.astype(float)*scale).clip(0,255).astype(np.uint8);bl=a.copy();bl[m]=(0.5*a[m]+0.5*pn[m]).astype(np.uint8)
 out=np.hstack([a,pn,bl]);cv2.imwrite(f'/work/scratch/renders/correct_{i}_{name}.jpg',cv2.cvtColor(out,cv2.COLOR_RGB2BGR));print(name,dx,dy,scale)
PY"
```
<details><summary>output</summary>

```
none -4.441067065073639 -0.2710882701206856 0.15714061227696746
fz 0.12883248089542576 -6.835664891599208 0.19047316709258583

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];F=len(T)
for a,b in [(0,160),(0,140),(900,1003),(920,1003)]:
 # rot avg
 Rm=np.mean(T[a:b,:3,:3],axis=0);u,s,v=np.linalg.svd(Rm);Rv=u@v
 if np.linalg.det(Rv)<0:u[:,-1]*=-1;Rv=u@v
 tm=np.median(T[a:b,:3,3],axis=0)
 da=np.array([np.degrees(Rot.from_matrix(Rv.T@T[i,:3,:3]).magnitude()) for i in range(a,b)])
 dt=np.linalg.norm(T[a:b,:3,3]-tm,axis=1)
 print(a,b,'rot deg med max',np.median(da),da.max(),'t med max cm',np.median(dt)*100,dt.max()*100,'pose',tm)
 # print orientation comp
 print(np.round(Rv,3))
PY"
```
<details><summary>output</summary>

```
0 160 rot deg med max 0.6816371205331679 2.270306007341136 t med max cm 1.4470105059444904 6.000398099422455 pose [-0.11663457  0.7856095   2.7336907 ]
[[-0.5   -0.865  0.041]
 [ 0.066  0.009  0.998]
 [-0.863  0.502  0.053]]
0 140 rot deg med max 0.6544004517883958 2.3528847547224334 t med max cm 1.6526224091649055 5.922354757785797 pose [-0.11683382  0.7850108   2.73295   ]
[[-0.499 -0.866  0.04 ]
 [ 0.067  0.007  0.998]
 [-0.864  0.501  0.055]]
900 1003 rot deg med max 0.6700047169750023 2.763254574698644 t med max cm 0.6229766178876162 2.5578945875167847 pose [0.38148212 0.795293   2.4139245 ]
[[ 0.881 -0.472  0.038]
 [ 0.02   0.117  0.993]
 [-0.473 -0.874  0.112]]
920 1003 rot deg med max 0.5550520916696119 1.6162690128090687 t med max cm 0.5674544721841812 2.3398395627737045 pose [0.3 …[+104 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
for a,b in [(0,160),(0,130),(900,1003),(930,1003)]:
 vis=o[a:b]&~p[a:b];freq=vis.mean(0);target=freq>.5
 print(a,b,'target',target.sum(),'vis mean',vis.sum((1,2)).mean(),'union',vis.any(0).sum(),'obj means',o[a:b].sum((1,2)).mean())
PY"
```
<details><summary>output</summary>

```
0 160 target 48292 vis mean 47663.2 union 49734 obj means 47726.91875
0 130 target 48318 vis mean 48127.21538461538 union 49730 obj means 48161.092307692306
900 1003 target 58221 vis mean 57985.330097087375 union 59891 obj means 58007.22330097087
930 1003 target 58282 vis mean 58183.24657534246 union 59217 obj means 58185.68493150685

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static.py <<'PY'
import numpy as np,trimesh,h5py,cv2,json
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];F=len(T)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=g.vertices[::50]
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
W,H=1536,1152
# mask raster

def predmask(R,t):
 vt=V@R.T+t;vc=vt@Tcw[:3,:3].T+Tcw[:3,3];uv=vc[:,:2]/vc[:,2,None]; uv=uv*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
 hull=cv2.convexHull(uv.astype(np.float32));m=np.zeros((H,W),np.uint8);cv2.fillPoly(m,[np.round(hull).astype(np.int32)],1);return m.astype(bool)
def rotavg(Rs):
 u,s,v=np.linalg.svd(Rs.mean(0));R=u@v
 if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@v
 return R
for a,b in [(0,150),(0,160),(900,1003),(920,1003)]:
 vis=o[a:b]&~p[a:b];target=vis.mean(0)>.5
 R0=rotavg(T[a:b,:3,:3]);t0=np.median(T[a:b,:3,3],0)
 def obj(x):
  R=Rotation.from_rotvec(x[:3]).as_matrix()@R0;t=t0+x[3:]
  m=predmask(R,t); return 1-(m&target).sum()/(m|target).sum()
 print('\\nphase',a,b,'base',obj(np.zeros(6)), 'base iou')
 x0=np.zeros(6);bounds=[(-.08,.08)]*3+[(-.1,.1),(-.1,.1),(-.18,.18)]
 res=minimize(obj,x0,method='Powell',bounds=bounds,options={'maxiter':200,'xtol':1e-5,'ftol':1e-7})
 print(res.fun,1 …[+406 chars]
```
<details><summary>output</summary>

```

phase 0 150 base 0.07915597595067592 base iou
0.02623258470513845 0.9737674152948615 [ 0.02217963 -0.07201511 -0.01001073 -0.01426941  0.02421768  0.05865956] success True 662
pose [[-0.4353951940967186, -0.8990856129410548, 0.04556432501907545], [0.09123201642317989, 0.0062857448217723, 0.9958098493697414], [-0.8956046832336786, 0.43772771717434483, 0.07928861470529655]] [-0.1308257441125926, 0.8095538140455654, 2.792165765489923]
delta cm [-1.42694126  2.42176772  5.86595628] rot deg 4.3553559871792
area 48447 48313 bbox (577, 639, 185, 304) (576, 639, 184, 304)

phase 0 160 base 0.07865534622534831 base iou
0.028525667018084455 0.9714743329819155 [ 0.00905285 -0.0725805  -0.00102965 -0.01306456  0.01877804  0.04283495] success True 524
pose [[-0.43625339608782626, -0.8990462641686192,  …[+1502 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_static.py';s=open(p).read()
s=s.replace(\"bounds=[(-.08,.08)]*3+[(-.1,.1),(-.1,.1),(-.18,.18)]\",\"bounds=[(0,0)]*3+[(-.1,.1),(-.1,.1),(-.18,.18)]\")
s=s.replace(\"x0=np.zeros(6)\",\"x0=np.zeros(6)\")
open('/work/scratch/opt_static_t.py','w').write(s)
PY
python /work/scratch/opt_static_t.py"
```
<details><summary>output</summary>

```

phase 0 150 base 0.07915597595067592 base iou
0.045165213865801745 0.9548347861341983 [ 0.          0.          0.         -0.02047531  0.03118832  0.10293966] success True 226
pose [[-0.4995657503604889, -0.8653340935707092, 0.04038553684949875], [0.0670304149389267, 0.007866737432777882, 0.997719943523407], [-0.8636787533760071, 0.5011337399482727, 0.0540737546980381]] [-0.13703164318439662, 0.8165244610718111, 2.836445860731828]
delta cm [-2.04753117  3.11883243 10.2939658 ] rot deg 0.0
area 48119 48313 bbox (576, 641, 186, 299) (576, 639, 184, 304)

phase 0 160 base 0.07865534622534831 base iou
0.04494427558257341 0.9550557244174266 [ 0.          0.          0.         -0.01699802  0.03443837  0.10150836] success True 218
pose [[-0.5002272725105286, -0.8649387359619141, 0.040664993226 …[+1435 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,im=cap.read();g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# crop object region ROI 500:1000,500:850
roi=g[550:1000,500:850]
for t in [30,40,50,60,70,80]:
 m=(roi<t).astype(np.uint8);n,lab,stats,c=cv2.connectedComponentsWithStats(m)
 inds=np.argsort(stats[1:,4])[::-1][:5]+1
 print('thr',t,[(stats[i].tolist(),c[i].tolist()) for i in inds])
PY"
```
<details><summary>output</summary>

```
thr 30 [([74, 96, 186, 309, 45516], [168.0800817295017, 246.32441339309253]), ([276, 14, 74, 251, 9780], [319.7652351738241, 155.05766871165645]), ([0, 40, 87, 82, 2877], [33.34966979492527, 80.10740354535974]), ([145, 17, 69, 70, 506], [176.65217391304347, 69.03754940711462]), ([221, 0, 66, 121, 310], [259.8451612903226, 59.903225806451616])]
thr 40 [([74, 96, 186, 313, 46334], [168.34877196011567, 247.5494669141451]), ([276, 10, 74, 255, 10224], [319.61962050078245, 156.23268779342723]), ([0, 32, 99, 90, 3329], [35.009011715229796, 80.47281465905678]), ([145, 0, 142, 121, 1517], [211.19446275543837, 54.238628872775216]), ([93, 75, 58, 19, 287], [124.32752613240419, 85.10452961672473])]
thr 50 [([74, 95, 186, 318, 47121], [168.14418199953312, 249.44684959996604]), ([275, 10, 75, 256, 1050 …[+1276 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 print(list(f.keys()));
 for k in f.keys():
  x=f[k];print(k,x.shape,x.dtype, x[0].min(),x[0].max(),x[0].mean())
PY"
```
<details><summary>output</summary>

```
['aligned', 'alignment', 'frame_names', 'raw']
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
AttributeError: 'Group' object has no attribute 'shape'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for root in ['aligned','alignment','raw']:
  g=f[root];print(root,g);print(dict(g.attrs));
  def v(name,obj): print(name,type(obj),getattr(obj,'shape',None),getattr(obj,'dtype',None),dict(obj.attrs))
  g.visititems(v)
  print('keys',list(g.keys())[:20])
PY"
```
<details><summary>output</summary>

```
aligned <HDF5 group "/aligned" (1 members)>
{}
front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (1003,) object {}
keys ['front_stereo_camera_left']
alignment <HDF5 group "/alignment" (1 members)>
{}
front_stereo_camera_left <class 'h5py._hl.group.Group'> None None {}
front_stereo_camera_left/scale <class 'h5py._hl.dataset.Dataset'> (1003,) float32 {}
front_stereo_camera_left/shift <class 'h5py._hl.dataset.Dataset'> (1003,) float32 {}
front_stereo_camera_left/valid_count <class 'h5py._hl.dataset.Dataset'> (1003,) int32 {}
keys ['front_stereo_camera_left']
raw <HDF5 group "/raw" (1 members)>
{}
front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (1003,) object {}
keys ['front_stereo_camera_left']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for k in ['aligned/raw']:
  pass
 a=f['aligned/front_stereo_camera_left'];r=f['raw/front_stereo_camera_left'];
 for i in [0,1]:
  for n,d in [('a',a),('r',r)]:
   v=d[i];print(i,n,type(v),getattr(v,'shape',None),getattr(v,'dtype',None),v if np.isscalar(v) else (v[:5] if hasattr(v,'__len__') else v))
 print('attrs',dict(a.attrs),dict(r.attrs))
PY"
```
<details><summary>output</summary>

```
0 a <class 'numpy.ndarray'> (633639,) uint8 [137  80  78  71  13]
0 r <class 'numpy.ndarray'> (592717,) uint8 [137  80  78  71  13]
1 a <class 'numpy.ndarray'> (633489,) uint8 [137  80  78  71  13]
1 r <class 'numpy.ndarray'> (591316,) uint8 [137  80  78  71  13]
attrs {} {}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2,pickle
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for grp in ['aligned','raw']:
  b=f[grp+'/front_stereo_camera_left'][0]; im=cv2.imdecode(b,cv2.IMREAD_UNCHANGED);print(grp,im.shape,im.dtype,im.min(),np.median(im),im.max())
print(pickle.load(open('/pipeline/depth/moge2/intrinsics.pkl','rb')))
PY"
```
<details><summary>output</summary>

```
aligned (1152, 1536) uint16 1726 3913.0 9458
raw (1152, 1536) uint16 1601 3629.0 8771
{'fx': 834.7869873046875, 'fy': 834.7869873046875, 'cx': 768.0, 'cy': 576.0, 'focals': [834.7869873046875, 834.8538818359375, 830.271484375, 831.2894287109375, 827.5482788085938, 828.1295166015625, 830.7008056640625, 828.8395385742188, 829.4880981445312, 829.7559814453125, 828.0789794921875, 827.2691650390625, 825.694580078125, 823.913818359375, 826.803955078125, 828.7825927734375, 832.4571533203125, 829.1350708007812, 825.6419677734375, 826.5977783203125, 822.7190551757812, 824.4073486328125, 822.284912109375, 824.5083618164062, 825.9105834960938, 828.5859375, 825.3428955078125, 823.3948974609375, 824.262451171875, 821.7535400390625, 822.8036499023438, 824.0311889648438, 825.3797607421875, 818.5206298828 …[+102091 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/depth_compare.py <<'PY'
exec(open('/work/scratch/edge_score.py').read().split(\"frames=\")[0])
import h5py,cv2,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: ad=f['aligned/front_stereo_camera_left']
frames=[0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
for i in frames:
 dep,norm,mask=render_geom(T[i]);b=ad[i];mo=cv2.imdecode(b,cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
 vis=om[i]&~pm[i];e=mask&vis
 # at same pixels
 md=mo[e];pd=dep[e]
 print(i,'moge med/mean',round(float(np.median(md)),3),round(float(md.mean()),3),'mesh z med',round(float(np.median(pd)),3),'diff med',round(float(np.median(md-pd)),3),'ratio',round(float(np.median(md)/np.median(pd)),4),'n',e.sum())
PY
python /work/scratch/depth_compare.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_compare.py", line 6, in <module>
    dep,norm,mask=render_geom(T[i]);b=ad[i];mo=cv2.imdecode(b,cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
                                      ~~^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 870, in __getitem__
    if self._fast_read_ok and (new_dtype is None):
       ^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/base.py", line 534, in __get__
    value = obj.__dict__[self.func.__name__] = self.func(obj)
                                               ^^^^^^^^^^^^^^
  File "/opt/conda/lib/p …[+969 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/depth_compare.py';s=open(p).read();s=s.replace(\"with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: ad=f['aligned/front_stereo_camera_left']\\nframes=\", \"frames=\")
s=s.replace(\"for i in frames:\\n dep,norm,mask=render_geom(T[i]);b=ad[i];\", \"for i in frames:\\n with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: b=f['aligned/front_stereo_camera_left'][i]\\n dep,norm,mask=render_geom(T[i]);\")
open(p,'w').write(s)
PY
python /work/scratch/depth_compare.py"
```
<details><summary>output</summary>

```
0 moge med/mean 2.511 2.545 mesh z med 0.233 diff med 2.278 ratio 10.7685 n 48271
50 moge med/mean 2.581 2.607 mesh z med 0.239 diff med 2.342 ratio 10.7868 n 46739
100 moge med/mean 2.594 2.623 mesh z med 0.235 diff med 2.359 ratio 11.0237 n 47339
150 moge med/mean 2.59 2.618 mesh z med 0.237 diff med 2.353 ratio 10.9379 n 44639
200 moge med/mean 2.747 2.756 mesh z med 0.247 diff med 2.498 ratio 11.1047 n 40573
250 moge med/mean 2.83 2.842 mesh z med 0.261 diff med 2.569 ratio 10.8347 n 35065
300 moge med/mean 3.026 3.112 mesh z med 0.272 diff med 2.754 ratio 11.1129 n 26492
350 moge med/mean 3.133 3.204 mesh z med 0.284 diff med 2.849 ratio 11.0172 n 26399
400 moge med/mean 2.996 3.09 mesh z med 0.274 diff med 2.722 ratio 10.9302 n 27759
450 moge med/mean 3.108 3.18 mesh z med 0.291 diff …[+952 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# compute euclid depths for frame0 via faster direct projection/raster logic
exec(open('/work/scratch/edge_score.py').read().split(\"frames=\")[0])
import h5py,cv2,numpy as np
for i in [0,50,100,200,300,400,500,600,700,800,900,950,1000]:
 Tmt=torch.tensor(T[i],dtype=torch.float32,device=device);vt=V@Tmt[:3,:3].T+Tmt[:3,3];Tcwt=torch.tensor(Tcw,device=device);vc=vt@Tcwt[:3,:3].T+Tcwt[:3,3];zz=vc[:,2];u=K[0,0]*vc[:,0]/zz+K[0,2];v=K[1,1]*vc[:,1]/zz+K[1,2];clip=torch.stack([2*u/W-1,2*v/H-1,zz/10,torch.ones_like(zz)],1)[None];rast,_=dr.rasterize(ctx,clip,Faces,resolution=[H,W]);depth=rast[0,...,2].cpu().numpy()*10;mask=rast[0,...,3].cpu().numpy()>0
 with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: mo=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
 valid=mask&om[i]&~pm[i];
 # erode a little
 valid=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))>0
 md=mo[valid];pd=depth[valid]
 print(i,'mo med',round(float(np.median(md)),3),'pz',round(float(np.median(pd)),3),'diff',round(float(np.median(md-pd)),3),'q',np.round(np.percentile(md-pd,[10,25,50,75,90]),3))
PY"
```
<details><summary>output</summary>

```
0 mo med 2.507 pz 2.329 diff 0.182 q [0.147 0.163 0.182 0.2   0.213]
50 mo med 2.576 pz 2.389 diff 0.191 q [0.155 0.172 0.191 0.21  0.223]
100 mo med 2.589 pz 2.349 diff 0.246 q [0.211 0.223 0.246 0.267 0.292]
200 mo med 2.742 pz 2.471 diff 0.269 q [0.179 0.236 0.269 0.311 0.336]
300 mo med 3.02 pz 2.718 diff 0.309 q [0.256 0.275 0.309 0.341 0.404]
400 mo med 2.989 pz 2.737 diff 0.258 q [0.214 0.232 0.258 0.281 0.313]
500 mo med 3.044 pz 2.683 diff 0.354 q [0.254 0.327 0.354 0.38  0.398]
600 mo med 2.849 pz 2.58 diff 0.263 q [0.199 0.213 0.263 0.287 0.299]
700 mo med 2.823 pz 2.595 diff 0.222 q [0.188 0.201 0.222 0.238 0.259]
800 mo med 2.629 pz 2.533 diff 0.087 q [0.03  0.062 0.087 0.095 0.105]
900 mo med 2.404 pz 2.108 diff 0.296 q [0.247 0.265 0.296 0.323 0.358]
950 mo med 2.338 pz 2.09 …[+119 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/frame0_reference_human.npz');print(z.files);print({k:z[k].shape for k in z.files})
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,cv2,h5py
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
V=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];vc=V@Tcw[:3,:3].T+Tcw[:3,3];z=vc[:,2];u=(K[0,0]*vc[:,0]/z+K[0,2]).round().astype(int);v=(K[1,1]*vc[:,1]/z+K[1,2]).round().astype(int)
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:mo=cv2.imdecode(f['aligned/front_stereo_camera_left'][0],cv2.IMREAD_UNCHANGED).astype(float)/1000
with h5py.File('/work/inputs/masks.h5','r') as f:pm=f['person'][0]
valid=(u>=0)&(u<1536)&(v>=0)&(v<1152);u=u[valid];v=v[valid];z=z[valid]
# group by pixel, min z
pix={}
for ui,vi,zi in zip(u,v,z):
 k=(vi,ui);pix[k]=min(pix.get(k,99),zi)
arr=np.array([(k[1],k[0],zi) for k,zi in pix.items()])
inper=pm[arr[:,1].astype(int),arr[:,0].astype(int)]
arr=arr[inper];md=mo[arr[:,1].astype(int),arr[:,0].astype(int)];zd=arr[:,2]
d=md-zd
print('n',len(d),'true human camera z med',np.median(zd),'moge med',np.median(md),'diff',np.percentile(d,[5,10,25,50,75,90,95]),'mean',d.mean())
# restrict depth pixels not at boundary perhaps and residual by image regions
for condname,cond in [('all',np.ones(len(d),bool)),('torso',(arr[:,1]>400)&(arr[:,1]<750)),('legs',(arr[:,1]>700)),('frontdepth',zd<np.percentile(zd,50))]:
 print(condname,len(d[cond]),np.percentile(d[cond],[10,50,90]))
PY"
```
<details><summary>output</summary>

```
n 7110 true human camera z med 3.3804857029055815 moge med 3.26 diff [-0.37545672 -0.33307936 -0.18461988 -0.08365736  0.00663808  0.11306248
  0.15684348] mean -0.07880698677257898
all 7110 [-0.33307936 -0.08365736  0.11306248]
torso 6147 [-0.3440843  -0.09885793  0.01663537]
legs 1361 [-0.04479722  0.09998857  0.16517851]
frontdepth 3555 [-0.11377648 -0.04491493  0.10687289]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/edge_trans.py <<'PY'
exec(open('/work/scratch/edge_score.py').read().split(\"frames=\")[0])
import numpy as np,cv2,json
from scipy.ndimage import distance_transform_edt
# contour-based actual image edge fit
def get_actual_edge(im,valid):
 gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
 # enhance local edges
 e=cv2.Canny(gray,40,100)>0
 near=cv2.dilate(valid.astype(np.uint8),np.ones((15,15),np.uint8))>0
 return e&near
def score_pose(Twm,im,valid):
 # render full mesh mask and actual edge DT
 rgb,mask,_=render(Twm)
 cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 pc=np.zeros_like(mask,np.uint8);cv2.drawContours(pc,cnts,-1,1,2);pc=pc>0
 ae=get_actual_edge(im,valid)&~pm[int(np.where(valid)[0][0])] if False else get_actual_edge(im,valid)
 # do not penalize contour in person mask
 pc=pc&~pm[0] if False else pc
 dta=distance_transform_edt(~ae);dtp=distance_transform_edt(~pc)
 return dta[pc].mean(),dtp[ae].mean(),ae.sum(),pc.sum()
frames=[0,30,60,950,980,1000]
for i in frames:
 im=get_frame(i);valid=om[i]&~pm[i]
 print('FRAME',i)
 for name,Tw in [('base',T[i]),('maskopt',None)]:
  if Tw is not None:
   rgb,mask,_=render(Tw);# filter contour person
   cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);pc=np.zeros_like(mask,np.uint8);cv2.drawContours(pc,cnts,-1,1,2);pc=pc>0
   ae=get_actual_edge(im,valid);pc=pc&~pm[i]
   dta=distance_transform_edt(~ae);dtp=distance_transform_edt( …[+106 chars]
```
<details><summary>output</summary>

```
FRAME 0
base 1.6366425424670736 5.479501797533991 2789 3076
FRAME 30
base 1.305455484934648 8.104817308045606 2763 2444
FRAME 60
base 3.0800472399502246 8.000162371117778 2845 2768
FRAME 950
base 4.80024615150528 17.967844051871722 3593 3026
FRAME 980
base 4.301523143981549 16.285287603080118 3567 3096
FRAME 1000
base 3.0998910461844433 17.003777332333833 3748 3067

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static_edge.py <<'PY'
import numpy as np,trimesh,h5py,cv2,json
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];F=len(T)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices[::50]
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
W,H=1536,1152
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def frame(i):cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
def rotavg(Rs):
 u,s,v=np.linalg.svd(Rs.mean(0));R=u@v
 if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@v
 return R
def contour(R,t):
 vt=V@R.T+t;vc=vt@Tcw[:3,:3].T+Tcw[:3,3];uv=vc[:,:2]/vc[:,2,None];uv=uv*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
 hull=cv2.convexHull(uv.astype(np.float32));m=np.zeros((H,W),np.uint8);cv2.fillPoly(m,[np.round(hull).astype(np.int32)],1)
 cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);pc=np.zeros_like(m);cv2.drawContours(pc,cnts,-1,1,1);return pc>0,m>0
for a,b,sel in [(0,110,list(range(0,70,10))),(920,1003,list(range(930,1003,10)))]:
 R0=rotavg(T[a:b,:3,:3]);t0=np.median(T[a:b,:3,3],0)
 items=[]
 for i in sel:
  im=frame(i);valid=o[i]&~p[i];gray=cv2.cvtColor(im,cv2.CO …[+998 chars]
```
<details><summary>output</summary>

```

phase 0 110 base 3.222767609517303
2.0482673470375308 [-0.00563558 -0.0179125  -0.00479067 -0.01937133  0.02162319  0.0459164 ] delta cm [-1.93713329  2.16231857  4.59163952] rot 1.1103676300141
0 p2 1.214495717571547 iou 0.9509136723477709
10 p2 1.0793362796139268 iou 0.9501629441220889
20 p2 1.260205237694671 iou 0.946925759840558
30 p2 1.7249372908460294 iou 0.9495158371940429
40 p2 1.6838327271665354 iou 0.9475682229244927

phase 920 1003 base 6.78254127109439
2.1421921461164417 [0.01451393 0.00654622 0.00581288 0.00464241 0.04200936 0.15518021] delta cm [ 0.46424088  4.20093554 15.51802119] rot 0.9711535628164991
930 p2 1.875136654319398 iou 0.9610252808988764
940 p2 1.9359482827803673 iou 0.958506571888412
950 p2 2.22616369977112 iou 0.9594635272082679
960 p2 1.7303471807073603 iou  …[+68 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);d=json.load(open('/work/inputs/ground_plane.json'))['plane'][3]
def stats(Tm):
 v=V@Tm[:3,:3].T+Tm[:3,3];s=v@n+d
 return np.percentile(s,[0,1,5,50,95,100])
for i in [0,50,100,150,900,950,980,1000]:print(i,stats(T[i]))
PY"
```
<details><summary>output</summary>

```
0 [-0.00823718  0.01992664  0.06181124  0.48172092  0.83657921  0.85070098]
50 [-0.02222465  0.00890754  0.05017469  0.47061159  0.82644486  0.83944824]
100 [-0.00892959  0.02007135  0.06067466  0.48113622  0.83653615  0.85037647]
150 [-0.01679102  0.01413764  0.0552995   0.47556047  0.8315758   0.84479253]
900 [-0.02182834  0.01243836  0.05240698  0.47451478  0.8323734   0.84543824]
950 [-0.01907119  0.01591378  0.0541228   0.47652559  0.83418195  0.84997445]
980 [-0.0175984   0.01771973  0.05567461  0.47802083  0.8358721   0.8518344 ]
1000 [-0.0198473   0.0161551   0.05371318  0.47614373  0.83412741  0.85119182]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# modify opt_static_t: free x,z only and fix y, no rot
p='/work/scratch/opt_static.py';s=open(p).read()
s=s.replace(\"bounds=[(-.08,.08)]*3+[(-.1,.1),(-.1,.1),(-.18,.18)]\",\"bounds=[(0,0)]*3+[(-.1,.1),(0,0),(-.25,.25)]\")
open('/work/scratch/opt_static_xz.py','w').write(s)
PY
python /work/scratch/opt_static_xz.py"
```
<details><summary>output</summary>

```

phase 0 150 base 0.07915597595067592 base iou
0.07534126535937535 0.9246587346406246 [ 0.          0.          0.         -0.01265797  0.          0.0363586 ] success True 126
pose [[-0.4995657503604889, -0.8653340935707092, 0.04038553684949875], [0.0670304149389267, 0.007866737432777882, 0.997719943523407], [-0.8636787533760071, 0.5011337399482727, 0.0540737546980381]] [-0.12921430201414788, 0.7853361368179321, 2.769864805907408]
delta cm [-1.26579705  0.          3.63586032] rot deg 0.0
area 50524 48313 bbox (575, 633, 190, 306) (576, 639, 184, 304)

phase 0 160 base 0.07865534622534831 base iou
0.07399890718913438 0.9260010928108656 [ 0.          0.          0.         -0.01230328  0.          0.03635025] success True 162
pose [[-0.5002272725105286, -0.8649387359619141, 0.0406649932265 …[+1477 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/edge_score.py').read().split(\"frames=\")[0])
import numpy as np
for i in [0,950,980,1000]:
 dep,norm,mask=render_geom(T[i]);ys,xs=np.where(mask);yo,xo=np.where(om[i]);print(i,'mesh',mask.sum(),xs.min(),xs.max(),ys.min(),ys.max(),'obj',om[i].sum(),xo.min(),xo.max(),yo.min(),yo.max(),'iouvisible',(mask&om[i]&~pm[i]).sum()/((mask|om[i])&~pm[i]).sum())
PY"
```
<details><summary>output</summary>

```
0 mesh 50625 574 766 634 947 obj 48579 576 759 638 942 iouvisible 0.9477352600475134
950 mesh 64089 737 953 643 995 obj 58207 740 938 651 975 iouvisible 0.9232465883846398
980 mesh 63820 738 953 642 993 obj 58517 740 943 651 974 iouvisible 0.9261072815149354
1000 mesh 63810 738 953 642 994 obj 58815 740 943 651 974 iouvisible 0.9305084745762712

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/edge_score.py').read().split(\"frames=\")[0])
import numpy as np
for i in [0,950]:
 dep,n,m=render_geom(T[i]);ys,xs=np.where(m);yo,xo=np.where(om[i]);print(i,'mesh c',xs.mean(),ys.mean(),'obj c',xo.mean(),yo.mean(),'delta',xo.mean()-xs.mean(),yo.mean()-ys.mean())
PY"
```
<details><summary>output</summary>

```
0 mesh c 669.5613432098766 781.682962962963 obj c 668.6982646822701 783.0368677823751 delta -0.8630785276064898 1.3539048194121506
950 mesh c 845.2347516734541 807.7233534615925 obj c 840.7936846083804 807.4522651914718 delta -4.441067065073639 -0.2710882701206856

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/depth_metrics.py <<'PY'
exec(open('/work/scratch/edge_score.py').read().split(\"frames=\")[0])
import h5py,cv2,numpy as np
frames=list(range(0,1003,20))
for i in frames:
 dep,norm,mask=render_geom(T[i]);valid=om[i]&~pm[i];visobj=valid
 # rendered visible mask excludes person
 mvis=mask&~pm[i]
 inter=(mvis&valid).sum();union=(mvis|valid).sum();iou=inter/union
 area_ratio=valid.sum()/max(mvis.sum(),1)
 with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:mo=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
 e=cv2.erode((mask&valid).astype(np.uint8),np.ones((5,5),np.uint8))>0;md=mo[e];pz=dep[e]
 ratio=np.median(md)/np.median(pz) if e.sum() else np.nan
 print(i,f'{iou:.4f}',f'{area_ratio:.4f}',f'{ratio:.4f}',valid.sum(),mvis.sum())
PY
python /work/scratch/depth_metrics.py"
```
<details><summary>output</summary>

```
0 0.9477 0.9596 10.7665 48579 50625
20 0.9563 0.9803 10.8858 47938 48902
40 0.9635 1.0040 10.7923 48239 48049
60 0.9206 0.9906 10.9948 48454 48916
80 0.9183 0.9923 11.0805 48404 48780
100 0.9372 0.9633 11.0196 48010 49839
120 0.9400 0.9455 10.9303 47663 50410
140 0.9437 0.9644 10.9336 45780 47468
160 0.9416 0.9687 10.9581 45332 46796
180 0.9304 0.9547 10.9563 43245 45297
200 0.9186 0.9563 11.0977 41425 43317
220 0.8927 0.9685 11.2853 38239 39481
240 0.9146 0.9736 10.9669 36668 37664
260 0.9186 0.9415 10.8682 33105 35162
280 0.9251 0.9429 10.9636 28931 30684
300 0.9454 0.9476 11.1100 26524 27991
320 0.9442 0.9472 11.0068 25964 27411
340 0.9456 0.9543 10.9633 26811 28095
360 0.9243 0.9786 11.0003 27222 27816
380 0.9355 0.9732 10.5925 26772 27510
400 0.9461 0.9496 10.9219 27813 29288
420 0.93 …[+1133 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_key.py <<'PY'
exec(open('/work/scratch/opt_static_edge.py').read().split(\"for a,b,sel\")[0])
from scipy.optimize import minimize
frames=[0,100,180,250,300,400,500,600,700,800,850,900,950,1000]
for i in frames:
 im=frame(i);valid=o[i]&~p[i];gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);e=cv2.Canny(gray,30,80)>0;near=cv2.dilate(valid.astype(np.uint8),np.ones((15,15),np.uint8))>0;e=e&near&~p[i];dt=distance_transform_edt(~e)
 R=T[i,:3,:3];t0=T[i,:3,3]
 def obj(x):
  t=t0+x;pc,m=contour(R,t);p2=dt[pc].mean();iou=(m&valid).sum()/(m|valid).sum();reg=np.linalg.norm(x)**2*10
  return p2+10*(1-iou)+reg
 res=minimize(obj,np.zeros(3),method='Powell',bounds=[(-.15,.15)]*3,options={'maxiter':60,'xtol':1e-4,'ftol':1e-5})
 pc,m=contour(R,t0+res.x);print(i,'base',round(obj(np.zeros(3)),3),'opt',round(res.fun,3),'delta_cm',np.round(res.x*100,2),'p2',round(dt[pc].mean(),2),'iou',round((m&valid).sum()/(m|valid).sum(),3),'floorchange',round(np.dot([-.018573,-.998367,-.054013],res.x),3))
PY
python /work/scratch/refine_key.py"
```
<details><summary>output</summary>

```
0 base 2.437 opt 1.767 delta_cm [-0.89  2.22  7.04] p2 1.25 iou 0.954 floorchange -0.026
100 base 6.337 opt 4.677 delta_cm [-1.58  2.8   6.72] p2 4.07 iou 0.945 floorchange -0.031
180 base 20.947 opt 19.325 delta_cm [-1.83  3.85  7.67] p2 18.16 iou 0.891 floorchange -0.042
250 base 7.503 opt 4.521 delta_cm [-0.89  3.47 14.34] p2 3.46 iou 0.916 floorchange -0.042
300 base 7.97 opt 5.559 delta_cm [ 0.04  3.95 13.19] p2 4.52 iou 0.915 floorchange -0.047
400 base 3.512 opt 2.426 delta_cm [0.38 3.12 8.51] p2 1.74 iou 0.94 floorchange -0.036
500 base 4.775 opt 2.554 delta_cm [-0.02  4.62 13.79] p2 1.54 iou 0.919 floorchange -0.054
600 base 2.695 opt 1.803 delta_cm [0.   3.59 8.01] p2 1.07 iou 0.934 floorchange -0.04
700 base 3.659 opt 3.105 delta_cm [-0.18  2.88  7.85] p2 2.52 iou 0.949 floorcha …[+463 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_floor.py <<'PY'
exec(open('/work/scratch/opt_static_edge.py').read().split(\"for a,b,sel\")[0])
from scipy.optimize import minimize
n=np.array([-.0185733148,-.9983674705,-.0540131998])
for i in [0,950,1000]:
 im=frame(i);valid=o[i]&~p[i];gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);e=cv2.Canny(gray,30,80)>0;near=cv2.dilate(valid.astype(np.uint8),np.ones((15,15),np.uint8))>0;e=e&near&~p[i];dt=distance_transform_edt(~e);R=T[i,:3,:3];t0=T[i,:3,3]
 def pose(x):
  dx,dz=x;dy=-(n[0]*dx+n[2]*dz)/n[1];return t0+np.array([dx,dy,dz])
 def obj(x):
  t=pose(x);pc,m=contour(R,t);p2=dt[pc].mean();iou=(m&valid).sum()/(m|valid).sum();return p2+10*(1-iou)+np.linalg.norm(x)**2*10
 res=minimize(obj,np.zeros(2),method='Powell',bounds=[(-.15,.15),(-.15,.25)],options={'maxiter':100,'xtol':1e-5,'ftol':1e-7})
 t=pose(res.x);pc,m=contour(R,t);print(i,'base',obj(np.zeros(2)),'opt',res.fun,'delta cm',(t-t0)*100,'p2',dt[pc].mean(),'iou',(m&valid).sum()/(m|valid).sum())
PY
python /work/scratch/refine_floor.py"
```
<details><summary>output</summary>

```
0 base 2.4374095962777345 opt 2.333800105578597 delta cm [-0.40176645  0.00201527  0.1009042 ] p2 1.515114408059979 iou 0.9181485900422325
950 base 7.202175083925482 opt 4.923428765463552 delta cm [-0.80914333 -0.31208113  6.04667236] p2 3.9277248939451455 iou 0.9041513088023989
1000 base 5.962091058808053 opt 3.7365189747978866 delta cm [-0.81023272 -0.27655254  5.39034468] p2 2.8516000239136807 iou 0.9144793341973362

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/overlay_opts.py <<'PY'
exec(open('/work/scratch/refine_key.py').read().split(\"frames=\")[0])
from scipy.optimize import minimize
import cv2,numpy as np
for i,delt in [(0,[-.0089,.0222,.0704]),(950,[-.0057,.038,.1351]),(1000,[-.0049,.032,.1334])]:
 im=frame(i);valid=o[i]&~p[i];gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);e=cv2.Canny(gray,30,80)>0;near=cv2.dilate(valid.astype(np.uint8),np.ones((15,15),np.uint8))>0;e=e&near&~p[i];dt=distance_transform_edt(~e);R=T[i,:3,:3]
 for name,t in [('base',T[i,:3,3]),('free',T[i,:3,3]+np.array(delt))]:
  pc,m=contour(R,t);out=im.copy();cnts,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);cv2.drawContours(out,cnts,-1,(255,0,0),4)
  # crop
  ys,xs=np.where(valid);x0=max(0,xs.min()-40);x1=min(W,xs.max()+40);y0=max(0,ys.min()-40);y1=min(H,ys.max()+40);cr=out[y0:y1,x0:x1];cr=cv2.resize(cr,(400,500));cv2.putText(cr,name,(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);cv2.imwrite(f'/work/scratch/ov_{i}_{name}.jpg',cv2.cvtColor(cr,cv2.COLOR_RGB2BGR))
PY
python /work/scratch/overlay_opts.py
python - <<'PY'
from PIL import Image,ImageDraw
for i in [0,950,1000]:
 a=Image.open(f'/work/scratch/ov_{i}_base.jpg');b=Image.open(f'/work/scratch/ov_{i}_free.jpg');sh=Image.new('RGB',(800,500));sh.paste(a,(0,0));sh.paste(b,(400,0));sh.save(f'/work/scratch/ovcomp_{i}.jpg')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/outputs
cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Trajectory recovery for the big black box episode.

The released reconstruction is used as the motion/rotation prior.  This solver
adds the temporal structure that a per-frame tracker misses:

* two visually verified rest phases are represented by one pooled rigid pose;
* rotations are averaged in SO(3) on each rest phase and gently smoothed while
  the object is manipulated;
* translations at sparse, deterministic keyframes are re-registered against
  the actual image silhouette (Canny edges + SAM object mask), pooling evidence
  over the static phases and interpolating a smooth absolute trajectory while
  manipulated.

Only files below --inputs and the object mesh supplied there are required.
No network access, random sampling, or evaluation annotations are used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt, median_filter
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# Scene boundaries were checked on the video and against the baseline speed
# profile.  The box is untouched through frame 159, is manipulated from 160
# through 899, and is again at rest from frame 900 onward (0-based indi …[+12681 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c '/usr/bin/time -v python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c 'time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/outputs/solve.py", line 379, in <module>
    main()
  File "/work/outputs/solve.py", line 352, in main
    interp = PchipInterpolator(motion_times, motion_values[:, axis], extrapolate=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/interpolate/_cubic.py", line 240, in __init__
    x, _, y, axis, _ = prepare_input(x, y, axis)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/interpolate/_cubic.py", line 63, in prepare_input
    raise ValueError("`x` must be strictly increasing sequence.")
ValueError: `x` must be strictly increasing sequence.

real	1m31.076s
user	1m45.693s
sys	0m14.516s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace('if frame_idx < b0:\\n            phase = 0','if frame_idx <= b0:\\n            phase = 0');open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py"
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
wrote /work/outputs/object_poses_world.npz with 1003 poses in 92.6s
static translations: [[-0.1380854688495571, 0.8083066518862856, 2.7948598140507497], [0.37911083040256044, 0.8329485650371036, 2.5623262645625795]]
motion translation key count: 73

real	1m33.769s
user	1m49.585s
sys	0m15.658s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/evaluate.py <<'PY'
import numpy as np,json,h5py,cv2,trimesh
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
base=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
res=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float)
F=len(base)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices[::50];Vf=g.vertices[::4]
plane=json.load(open('/work/inputs/ground_plane.json'))['plane'];n=np.array(plane[:3]);d=plane[3]
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
W,H=1536,1152
cap=cv2.VideoCapture('/work/inputs/video.mp4')
segs=[(0,160),(160,520),(520,850),(850,900),(900,1003)]

def hullmask(Tm):
 vc=(V@Tm[:3,:3].T+Tm[:3,3])@Tcw[:3,:3].T+Tcw[:3,3];uv=vc[:,:2]/vc[:,2,None];uv=uv*[K[0,0],K[1,1]]+[K[0,2],K[1,2]];h=cv2.convexHull(uv.astype(np.float32));m=np.zeros((H,W),np.uint8);cv2.fillPoly(m,[np.round(h).astype(np.int32)],1);return m.astype(bool)
def metrics(T):
 ious=[];p2s=[];bboxes=[]
 for i in range(F):
  ok,im=cap.read();
  if i==0: pass
  vis=o[i]&~p[i];m=hullmask(T[i]);inter=(m&vis).sum();union=(m|vis).sum();ious.append(inter/max(union,1))
  if i%5==0:
   gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);e=cv2.Canny(gray,30,80)>0;near=cv2.dilate(vis.astype( …[+2421 chars]
```
<details><summary>output</summary>

```

 baseline
overall IoU mean med min 0.8790488878695872 0.8749944013974112 0.7951330414280903
contour edge mean med p90 4.811652467850252 4.3480841077496395 7.279553740518577
rot step mean med p95 max 0.6818190221982775 0.5190234622691781 1.7298527140008797 4.93722017711092
t step cm mean med p95 max 1.4139825247994866 0.9395371384148258 4.3628731197143455 10.852502937990636
floor min distance percentiles cm [-2.23668163 -1.67910243 -0.75650043  0.75970342  3.91144546]
static spread deg med/max, cm med/max [(0.6816368954425305, 2.270307111577724, 1.4470113052240023, 6.000409849699447), (0.670005683207167, 2.763252302238403, 0.6229766175351432, 2.557894676595236)]
0 160 iou 0.9079755334146695 edge 6.484187732685329 rstep 0.4450370377483577 tstep cm 0.89227969301274
160 520 iou 0.870188441185 …[+2670 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/edge_score.py').read().split(\"frames=\")[0])
import numpy as np
res=np.load('/work/outputs/object_views.npz') if False else np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for i in [0,950,1000]:
 for nm,Tm in [('base',T[i]),('res',res[i])]:
  dep,norm,mask=render_geom(Tm);ys,xs=np.where(mask);print(i,nm,mask.sum(),xs.min(),xs.max(),ys.min(),ys.max())
 ys,xs=np.where(om[i]);print('obj',om[i].sum(),xs.min(),xs.max(),ys.min(),ys.max())
PY"
```
<details><summary>output</summary>

```
0 base 50625 574 766 634 947
0 res 47637 573 761 639 942
obj 48579 576 759 638 942
950 base 64089 737 953 643 995
950 res 56280 736 938 651 980
obj 58207 740 938 651 975
1000 base 63810 738 953 642 994
1000 res 56280 736 938 651 980
obj 58815 740 943 651 974

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/result_overlay.py <<'PY'
import numpy as np,json,h5py,cv2,trimesh
from PIL import Image,ImageDraw
import os
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices[::50];F=len(T)
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');frames=list(range(0,1003,25))
os.makedirs('/work/scratch/resov',exist_ok=True)
for i in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read()
 vc=(V@T[i,:3,:3].T+T[i,:3,3])@Tcw[:3,:3].T+Tcw[:3,3];uv=vc[:,:2]/vc[:,2,None];uv=uv*[K[0,0],K[1,1]]+[K[0,2],K[1,2]];h=cv2.convexHull(uv.astype(np.float32));cv2.polylines(im,[np.round(h).astype(np.int32)],True,(0,0,255),5)
 ys,xs=np.where(o[i]);x0=max(0,xs.min()-50);x1=min(1536,xs.max()+50);y0=max(0,ys.min()-50);y1=min(1152,ys.max()+50);cr=im[y0:y1,x0:x1];cr=cv2.resize(cr,(360,420));cv2.putText(cr,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);cv2.imwrite(f'/work/scratch/resov/{i:04d}.jpg',cr)
fs=sorted(os.listdir('/work/scratch/resov'));sh=Image.new('RGB',(360*5,420*9))
for j,f in enumerate(fs):sh.paste(Image.open('/work/scratch/resov/'+f),((j%5)*360,(j//5)*420))
sh.save('/work/scratch/result_contact.jpg',quality=92)
PY
python /work/scratch/result_overlay.p …[+2 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sha256sum /work/inputs/object_mesh.glb /work/inputs/reconstruction.npz /work/inputs/masks.h5 /work/inputs/video.mp4; python - <<'PY'
import numpy as np
p='/work/outputs/object_poses_world.npz';z=np.load(p);print(z.files,z['T_world_mesh'].shape,z['T_world_mesh'].dtype,np.isfinite(z['T_world_mesh']).all());
# validate bottom row and det
T=z['T_world_mesh'];print('bottom maxerr',np.abs(T[:,3]-[0,0,0,1]).max(),'det range',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
PY"
```
<details><summary>output</summary>

```
b6814ef943adf81d5013b1332e6c3b607e9c80d713b7d70c5a6d0ba602df1e27  /work/inputs/reconstruction.npz
b3866396e93476e2789b7366935bbc8cd77b711305c8e225b033c55014dcaffb  /work/inputs/masks.h5
9fe5b6797fe91a3a696e70ab86c0500a6e238330cf7916c0fe86c21e37fce3b2  /work/inputs/video.mp4
['T_world_mesh'] (1003, 4, 4) float64 True
bottom maxerr 0.0 det range 0.9999999999999991 1.000000000000001

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys, numpy, scipy, cv2, trimesh, h5py, torch, nvdiffrast, PIL, skimage
print('python',sys.version.split()[0])
for m in [numpy,scipy,cv2,trimesh,h5py,torch,nvdiffrast,PIL,skimage]: print(m.__name__,getattr(m,'__version__','unknown'),getattr(m,'__file__',''))
PY"
```
<details><summary>output</summary>

```
python 3.11.10
numpy 1.26.3 /opt/conda/lib/python3.11/site-packages/numpy/__init__.py
scipy 1.16.3 /opt/conda/lib/python3.11/site-packages/scipy/__init__.py
cv2 4.11.0 /opt/conda/lib/python3.11/site-packages/cv2/__init__.py
trimesh 5.1.1 /opt/conda/lib/python3.11/site-packages/trimesh/__init__.py
h5py 3.16.0 /opt/conda/lib/python3.11/site-packages/h5py/__init__.py
torch 2.5.1+cu124 /opt/conda/lib/python3.11/site-packages/torch/__init__.py
nvdiffrast 0.4.0 /opt/conda/lib/python3.11/site-packages/nvdiffrast/__init__.py
PIL 10.2.0 /opt/conda/lib/python3.11/site-packages/PIL/__init__.py
skimage 0.26.0 /opt/conda/lib/python3.11/site-packages/skimage/__init__.py

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "du -sh /work/outputs; ls -lh /work/outputs; python - <<'PY'
# MoGe disagreement for final report: median object-depth prediction minus baseline/result at sampled frames
import numpy as np,h5py,cv2,json,trimesh
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'];R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices[::100];F=len(B)
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
def pred(Tm):
 vc=(V@Tm[:3,:3].T+Tm[:3,3])@Tcw[:3,:3].T+Tcw[:3,3];z=vc[:,2];u=(K[0,0]*vc[:,0]/z+K[0,2]).round().astype(int);v=(K[1,1]*vc[:,1]/z+K[1,2]).round().astype(int);m=np.zeros((1152,1536),bool);ok=(u>=0)&(u<1536)&(v>=0)&(v<1152);m[v[ok],u[ok]]=True;return m,z
for a,b in [(0,160),(160,520),(520,850),(850,900),(900,1003)]:
 bb=[];rr=[]
 with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
  for i in range(a,b,20):
   mo=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
   for Tm,arr in [(B[i],bb),(R[i],rr)]:
    m,zp=pred(Tm);e=m&o[i]&~p[i];arr.append(np.median(mo[e]-zp[e])) if e.sum() else None
 print(a,b,'baseline MoGe - z object median cm',round(float(np.median(bb))*100,2),'result',round(float(np.median(rr))*100,2))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 13, in <module>
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,cv2,json,trimesh
rec=np.load('/work/inputs/reconstruction.npz');B=rec['T_world_mesh'];R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices[::50]
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
def pred(Tm):
 vc=(V@Tm[:3,:3].T+Tm[:3,3])@Tcw[:3,:3].T+Tcw[:3,3];zc=vc[:,2];u=(K[0,0]*vc[:,0]/zc+K[0,2]).round().astype(int);v=(K[1,1]*vc[:,1]/zc+K[1,2]).round().astype(int);m=np.zeros((1152,1536),bool);ok=(u>=0)&(u<1536)&(v>=0)&(v<1152);m[v[ok],u[ok]]=True;return m,zc
for a,b in [(0,160),(160,520),(520,850),(850,900),(900,1003)]:
 bb=[];rr=[]
 with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
  for i in range(a,b,20):
   mo=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
   for Tm,arr in [(B[i],bb),(R[i],rr)]:
    m,zc=pred(Tm);e=m&o[i]&~p[i]
    if e.sum():arr.append(np.median(mo[e]-zc[e]))
 print(a,b,'baseline',round(float(np.median(bb))*100,2),'result',round(float(np.median(rr))*100,2))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 13, in <module>
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/edge_score.py').read().split(\"frames=\")[0])
rec=np.load('/work/inputs/reconstruction.npz');B=rec['T_world_mesh'];R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for a,b in [(0,160),(160,520),(520,850),(850,900),(900,1003)]:
 bb=[];rr=[]
 with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
  for i in range(a,b,20):
   mo=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
   for Tm,arr in [(B[i],bb),(R[i],rr)]:
    dep,norm,mask=render_geom(Tm);e=cv2.erode((mask&om[i]&~pm[i]).astype(np.uint8),np.ones((5,5),np.uint8))>0
    if e.sum():arr.append(np.median(mo[e]-(dep[e]*10)))
 print(a,b,'baseline',round(float(np.median(bb))*100,2),'result',round(float(np.median(rr))*100,2))
PY"
```
<details><summary>output</summary>

```
0 160 baseline 21.98 result 15.3
160 520 baseline 26.89 result 10.66
520 850 baseline 26.28 result 12.8
850 900 baseline 26.18 result 8.72
900 1003 baseline 21.74 result 5.83

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -lh /work/outputs; test -f /work/outputs/REPORT.md && echo report_exists || true; test -f /work/outputs/TOOLS.md && echo tools_exists || true'
```
<details><summary>output</summary>

```
total 88K
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 20:21 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  67K Oct 20 20:23 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  14K Oct 20 20:21 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Big Black Box: World-Frame Trajectory Recovery

## Result

"'`object_poses_world.npz` contains `T_world_mesh` as `(1003, 4, 4)` `float64`.
Every transform is finite, has determinant 1 to numerical precision, and maps the
exact `object_mesh.glb` mesh coordinates into the released world frame.

The baseline rotation trajectory was retained as the rotation prior. I evaluated
textured renders of the baseline and of 180-degree alternatives about each local
principal axis. The alternatives reduced silhouette agreement and broke visible
handle/latch correspondence, so no global symmetry flip was applied. The main
gains came from pooling the two rest phases, removing isolated depth/tracker
noise, and re-registering translation against the actual image silhouette.

## Phases

| Index range | Phase | Evidence and treatment |
|---|---|---|
| 0-159 | Initial rest | Box upright and untouched; image silhouette and baseline pose are constant. One pooled rigid pose is used for all 160 frames. |
| 160-519 | Tip and roll | The person tips the box onto its side and rolls/moves it toward the camera. Rotation is kept close to the tracked motion; translation is re-registered and interpolated smoothly. |
| 520-849 | Manipulate on side | The box is reoriented while mostly on its broad side. Several rapid depth changes in the baseline are removed by keyframe registration and temporal interpolation. |
| 850-899 | Lift upright / set down | The bo …[+5543 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/TOOLS.md <<'MD'
# Tools Used

No external repository was cloned and no model weights or files were downloaded.
All tools below were already installed in the container. The final solver does
not use the network, "'`/pipeline`, or `/work/scratch`.

## Required by `solve.py`

| Tool | Version | Use |
|---|---:|---|
| Python | 3.11.10 | Runtime |
| NumPy | 1.26.3 | Arrays, linear algebra, deterministic numerical operations |
| SciPy | 1.16.3 | SO(3) utilities, Powell optimization, robust filters, PCHIP interpolation |
| OpenCV (`cv2`) | 4.11.0 | Video decoding, Canny edges, masks/contours, distance transforms |
| trimesh | 5.1.1 | Loads the supplied GLB and exposes exact mesh vertices |
| h5py | 3.16.0 | Reads supplied `masks.h5` |

## Used for experiments/validation, not required by the final solver

| Tool | Version | Use |
|---|---:|---|
| PyTorch | 2.5.1+cu124 | Differentiable textured-render experiments |
| nvdiffrast | 0.4.0 | Rasterized textured-mesh hypothesis rendering and depth/edge experiments; helped reject symmetry flips and quantify depth bias |
| Pillow | 10.2.0 | Contact sheets for visual inspection |
| scikit-image | 0.26.0 | Early photometric/structural score experiments |

## Existing read-only pipeline artifacts inspected

* CARI4D reconstruction and stage reports under `/pipeline`.
* Provided MoGe-2 aligned depth under `/pipeline/depth/moge2/aligned.h5`, used
  only as an independent validation cue. MoGe-2 model id
  `Rui …[+880 chars]
```
<details><summary>output</summary>

```
REPORT.md 6979 bytes
TOOLS.md 2248 bytes
object_poses_world.npz 68488 bytes
solve.cpython-311.pyc 21958 bytes
solve.py 13950 bytes

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
p=Path('/work/outputs/__pycache__/solve.cpython-311.pyc')
if p.exists(): p.unlink()
d=p.parent
if d.exists() and not any(d.iterdir()): d.rmdir()
PY
ls -lh /work/outputs"
```
<details><summary>output</summary>

```
total 96K
-rw-rw-r-- 1 hanzhou hanzhou 6.9K Oct 20 20:26 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 2.2K Oct 20 20:26 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou  67K Oct 20 20:23 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  14K Oct 20 20:21 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md'
s=open(p).read().replace('without inventing a pose flips.','without inventing pose flips.')
open(p,'w').write(s)
PY
python - <<'PY'
import numpy as np
p='/work/outputs/object_poses_world.npz'
z=np.load(p)
assert z.files == ['T_world_mesh']
T=z['T_world_mesh']
assert T.shape == (1003,4,4) and T.dtype == np.float64 and np.isfinite(T).all()
assert np.abs(T[:,3]-np.array([0,0,0,1])).max() < 1e-12
assert np.allclose(np.linalg.det(T[:,:3,:3]),1,atol=1e-10)
print('final output valid',T.shape,T.dtype)
PY"
```
<details><summary>output</summary>

```
final output valid (1003, 4, 4) float64

```
</details>

**Agent:** Completed the trajectory recovery task.

Outputs:
- `/work/outputs/object_poses_world.npz` — 1003 world-frame poses, `float64`, shape `(1003, 4, 4)`
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, method, validation metrics, confidence, runtime
- `/work/outputs/TOOLS.md` — tools, versions, provenance, input checksums

Validation highlights:
- Visible-mask IoU improved from `0.879` to `0.937`
- Contour-to-edge error improved from `4.81 px` to `3.17 px`
- Worst translation jump reduced from `10.85 cm` to `1.40 cm`
- Rest phases now use exactly constant rigid poses
- Final solver runtime: approximately `93 s`, CPU-only
