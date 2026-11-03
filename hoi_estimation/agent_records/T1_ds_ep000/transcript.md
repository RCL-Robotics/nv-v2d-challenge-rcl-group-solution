# T1_ds_ep000

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T15:04:19Z

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
/usr/bin/bash -c "find /work -maxdepth 2 -type f -printf '%p\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/work/.codex_home/.sandbox_migration
/work/.codex_home/config.toml
/work/.codex_home/goals_1.sqlite
/work/.codex_home/goals_1.sqlite-shm
/work/.codex_home/goals_1.sqlite-wal
/work/.codex_home/installation_id
/work/.codex_home/logs_2.sqlite
/work/.codex_home/logs_2.sqlite-shm
/work/.codex_home/logs_2.sqlite-wal
/work/.codex_home/memories_1.sqlite
/work/.codex_home/memories_1.sqlite-shm
/work/.codex_home/memories_1.sqlite-wal
/work/.codex_home/model_catalog.json
/work/.codex_home/queue_1.sqlite
/work/.codex_home/queue_1.sqlite-shm
/work/.codex_home/queue_1.sqlite-wal
/work/.codex_home/state_5.sqlite
/work/.codex_home/state_5.sqlite-shm
/work/.codex_home/state_5.sqlite-wal
/work/.codex_home/thread_history_1.sqlite
/work/.codex_home/thread_history_1.sqlite-shm
/work/.codex_home/thread_history_ …[+420 chars]
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
/usr/bin/bash -c "cat /work/inputs/task.txt; printf '\\n--- episode ---\\n'; cat /work/inputs/episode.json; printf '\\n--- camera ---\\n'; cat /work/inputs/camera.json; printf '\\n--- ground ---\\n'; cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
Walk over to the tall bar stool and using both hands, tip the stool from a face down lying down position back to upright.

--- episode ---
{
 "episode_index": 0,
 "object": "tall_bar_stool",
 "frames": 778
}
--- camera ---
{
 "K": [
  [
   819.9987182617188,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.9987182617188,
   560.5000610351562
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "K_pipeline": [
  [
   818.9199829101562,
   0.0,
   768.0
  ],
  [
   0.0,
   818.919921875,
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
   0.9994056724264703,
   -0.00630215050390631,
   0.03389077781375614,
   0.057152218309275395
  ],
  [
   0.005063151340571403,
   0.9993200613515929,
   0.036520945753789544,
   0.0034467307862450745
  ],
  [
   -0.03409789466078137,
   -0.036 …[+672 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, trimesh, os
p='/work/inputs/reconstruction.npz'
z=np.load(p)
for k in z.files:
 a=z[k]
 print(k, a.shape, a.dtype, 'min/max', (float(np.nanmin(a)),float(np.nanmax(a))) if np.issubdtype(a.dtype,np.number) else '')
with h5py.File('/work/inputs/masks.h5','r') as f:
 print('masks keys',list(f.keys()))
 for k in f: print(k, f[k].shape, f[k].dtype)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=m.to_geometry()
print('geometry type',type(g), 'bounds',g.bounds, 'vertices',len(g.vertices),'faces',len(g.faces),'visual',type(g.visual),'extents',g.extents)
PY"
```
<details><summary>output</summary>

```
T_world_mesh (778, 4, 4) float32 min/max (-0.7801939249038696, 3.243943929672241)
T_cam_mesh_foundationpose (778, 4, 4) float32 min/max (-0.7555374503135681, 3.3316242694854736)
T_cam_mesh_coconet (778, 4, 4) float32 min/max (-0.7507079839706421, 3.1650047302246094)
human_vertices_world (778, 4610, 3) float32 min/max (-1.0448040962219238, 4.033820152282715)
human_keypoints_world (778, 70, 3) float32 min/max (-1.0350888967514038, 4.026278972625732)
contact_prob_right_left (778, 2) float32 min/max (1.1246928735553219e-11, 1.0)
masks keys ['object', 'person']
object (778, 1152, 1536) bool
person (778, 1152, 1536) bool
geometry type <class 'trimesh.base.Trimesh'> bounds [[-0.20217043 -0.23005383 -0.35287058]
 [ 0.26516873  0.22619514  0.53656369]] vertices 51247 faces 95759 visual <class 'trim …[+79 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi; python - <<'PY'
import torch, cv2, nvdiffrast
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('cv2',cv2.__version__)
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124 cuda True NVIDIA RTX 6000 Ada Generation
cv2 4.11.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 /work/inputs/video.mp4'
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
duration=25.933333
nb_frames=778

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/contact && python - <<'PY'
import cv2, numpy as np, h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 om=f['object']; pm=f['person']
 idxs=list(range(0,778,39))
 for i in idxs:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
  if not ok: print('bad',i);continue
  ov=im.copy()
  o=om[i].astype(bool); p=pm[i].astype(bool)
  ov[o]=(0.35*ov[o]+0.65*np.array([0,255,0])).astype(np.uint8)
  ov[p]=(0.75*ov[p]+0.25*np.array([255,0,0])).astype(np.uint8)
  txt=f'{i}'
  cv2.rectangle(ov,(0,0),(150,55),(0,0,0),-1); cv2.putText(ov,txt,(10,42),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
  ov=cv2.resize(ov,(384,288))
  frames.append(ov)
cap.release()
# grid 4 cols
rows=[]
for j in range(0,len(frames),4):
 row=frames[j:j+4]
 while len(row)<4: row.append(np.zeros_like(frames[0]))
 rows.append(np.concatenate(row,axis=1))
out=np.concatenate(rows,axis=0)
cv2.imwrite('/work/scratch/contact/contact_masks.jpg',out)
print(out.shape)
# also raw
PY"
```
<details><summary>output</summary>

```
(1440, 1536, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh']; F=len(T)
# per-frame relative rotation magnitudes
Rr=T[:,:3,:3]
rel=R.from_matrix(np.einsum('nji,njk->nik',Rr[:-1],Rr[1:])).magnitude()*180/np.pi
dt=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)
with h5py.File('/work/inputs/masks.h5','r') as f:
 o=f['object'][:]; p=f['person'][:]
area=o.sum((1,2)); ar=(area>0).astype(int)
# components approximate bbox and centroid
bboxes=[]; cents=[]
for i in range(F):
 ys,xs=np.where(o[i])
 bboxes.append([xs.min(),ys.min(),xs.max(),ys.max()] if len(xs) else [0]*4)
 cents.append([xs.mean(),ys.mean()] if len(xs) else [0,0])
bboxes=np.array(bboxes); cents=np.array(cents)
np.set_printoptions(precision=3,suppress=True,linewidth=200)
print('trans first f0-10\\n',T[:11,:3,3])
print('trans samples every 20\\n',T[::20,:3,3])
print('rel rot deg every 20\\n',rel[19::20])
print('dt every 20\\n',dt[19::20])
print('max rel',rel.max(),np.argmax(rel),'max dt',dt.max(),np.argmax(dt))
print('area every 20\\n',area[::20])
print('cent every 20\\n',cents[::20])
print('bbox samples\\n',bboxes[::20])
print('contact every 20',z['contact_prob_right_left'][::20])
# frame static metrics T rot from first, positions dist to mode-ish
for i in range(0,F,20): print(i,'T',T[i,:3,3],'area',area[i],'cent',cents[i], 'rotrelprev', rel[i-1] if i else 0)
PY"
```
<details><summary>output</summary>

```
trans first f0-10
 [[-0.121  0.983  2.724]
 [-0.12   0.972  2.698]
 [-0.124  0.977  2.712]
 [-0.123  0.97   2.693]
 [-0.124  0.982  2.734]
 [-0.124  0.983  2.734]
 [-0.123  0.979  2.723]
 [-0.122  0.978  2.721]
 [-0.122  0.982  2.73 ]
 [-0.122  0.981  2.732]
 [-0.12   0.975  2.715]]
trans samples every 20
 [[-0.121  0.983  2.724]
 [-0.119  0.981  2.722]
 [-0.12   0.985  2.718]
 [-0.12   0.989  2.734]
 [-0.112  0.975  2.699]
 [-0.113  0.997  2.761]
 [-0.118  1.     2.775]
 [-0.104  1.011  2.809]
 [-0.111  1.007  2.801]
 [-0.11   1.004  2.787]
 [-0.101  0.984  2.747]
 [-0.098  0.982  2.735]
 [-0.102  0.988  2.744]
 [-0.106  0.986  2.745]
 [-0.113  0.98   2.753]
 [-0.135  0.929  2.757]
 [-0.109  0.852  2.741]
 [-0.078  0.818  2.761]
 [-0.057  0.804  2.809]
 [-0.037  0.785  2.837]
 [-0.006  0. …[+6920 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort | head -300"
```
<details><summary>output</summary>

```
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 974843416
/pipeline/depth/moge2/depth_report.json 1804
/pipeline/depth/moge2/intrinsics.pkl 82875
/pipeline/depth/moge2/raw.npy 2753298560
/pipeline/export/episode_000000/edex 559
/pipeline/export/episode_000000/wild_export.json 2778
/pipeline/foundationpose/episode_000000.pkl 222559
/pipeline/inference/coconet.pth 595065104
/pipeline/inference/episode_000000_step200000_before_after.mp4 57953578
/pipeline/inference/episode_000000_step200000_reconstruction.mp4 21223419
/pipeline/inference/refined.pth 595128736
/pipeline/mhr_init/episode_000000.pkl 3472357
/pipeline/mhr_init/episode_000000.sam3d.h5 782407
/pipeline/mhr_init/episode_000000.stats.pkl 7507
/pipeline/pipeline_report.json 5837
/pipeline/work/ …[+104 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
A=z['T_world_mesh']; B=z['T_cam_mesh_foundationpose']; C=z['T_cam_mesh_coconet']
for name,X in [('fp',B),('coco',C)]:
 # relative cam poses
 relR=R.from_matrix(np.einsum('nji,njk->nik',X[:,:3,:3],A[:,:3,:3])).magnitude()*180/np.pi
 # transpose because world->cam? Compare rotations invariant inverse
 print(name,'rot diff deg mean/med/max',relR.mean(),np.median(relR),relR.max())
 print('samples',relR[::50])
 for i in [0,100,200,300,400,500,600,700,777]: print(i,'A t',A[i,:3,3],'X t',X[i,:3,3],'rotdiff',relR[i])
# relative FP vs CoCo
rel=R.from_matrix(np.einsum('nji,njk->nik',B[:,:3,:3],C[:,:3,:3])).magnitude()*180/np.pi
print('fp-coco rot',rel.mean(),np.median(rel),rel.max(),rel[::50])
PY"
```
<details><summary>output</summary>

```
fp rot diff deg mean/med/max 4.561413001709871 4.257297549387468 6.8369699230805345
samples [3.80814734 4.22817572 3.95722503 3.83674314 3.26332595 3.99937261
 3.97210145 4.50085539 6.50676915 3.15304693 4.62246363 4.80423595
 5.48569204 6.46898436 5.93790861 6.46352752]
0 A t [-0.12149151  0.98271966  2.723595  ] X t [-0.30639973  0.8619293   2.8141682 ] rotdiff 3.8081473401730346
100 A t [-0.11252268  0.99662554  2.7605839 ] X t [-0.30588606  0.860295    2.810953  ] rotdiff 3.9572250259221047
200 A t [-0.10068303  0.9839784   2.747082  ] X t [-0.3046054   0.86516005  2.8115413 ] rotdiff 3.263325949100223
300 A t [-0.13521641  0.9289664   2.7568514 ] X t [-0.31499442  0.75997293  2.674074  ] rotdiff 3.972101447326008
400 A t [-0.00577883  0.7427543   2.856499  ] X t [-0.17383336  0.546526 …[+1993 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'find /code/reconstruction -maxdepth 3 -type f | sort | head -100'
```
<details><summary>output</summary>

```
/code/reconstruction/.cursorignore
/code/reconstruction/.dockerignore
/code/reconstruction/.gitattributes
/code/reconstruction/.gitignore
/code/reconstruction/.venv/.gitignore
/code/reconstruction/.venv/.lock
/code/reconstruction/.venv/CACHEDIR.TAG
/code/reconstruction/.venv/bin/activate
/code/reconstruction/.venv/bin/activate.bat
/code/reconstruction/.venv/bin/activate.csh
/code/reconstruction/.venv/bin/activate.fish
/code/reconstruction/.venv/bin/activate.nu
/code/reconstruction/.venv/bin/activate.ps1
/code/reconstruction/.venv/bin/activate.xsh
/code/reconstruction/.venv/bin/activate_this.py
/code/reconstruction/.venv/bin/deactivate.bat
/code/reconstruction/.venv/bin/f2py
/code/reconstruction/.venv/bin/flask
/code/reconstruction/.venv/bin/jp.py
/code/reconstruction/.venv/bin/numpy-config …[+4097 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/pipeline_report.json; printf '\\n--- depth report ---\\n'; cat /pipeline/depth/moge2/depth_report.json; printf '\\n--- export ---\\n'; cat /pipeline/export/episode_000000/wild_export.json"
```
<details><summary>output</summary>

```
{
  "checkpoint": {
    "mtime_ns": 1791121400000000000,
    "path": "/l/users/han.zhou/v2d/weights/cari4d/2026-08-25-09-35-57/step200000.pth",
    "repository": "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9",
    "sha256": "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3",
    "size": 2084484133,
    "step": 200000
  },
  "comparison_video": {
    "mtime_ns": 1791145941000000000,
    "path": "/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/inference/episode_000000_step200000_before_after.mp4",
    "size": 57953578
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frames": …[+9657 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, numpy as np
for p in ['/pipeline/foundationpose/episode_000000.pkl','/pipeline/mhr_init/episode_000000.pkl','/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 print('\\n',p)
 try:
  x=pickle.load(open(p,'rb'))
  print(type(x))
  if isinstance(x,dict):
   for k,v in x.items():
    if isinstance(v,np.ndarray): print(k,v.shape,v.dtype)
    elif isinstance(v,(list,tuple)): print(k,type(v),len(v))
    elif isinstance(v,dict): print(k,'dict',list(v)[:20])
    else: print(k,type(v),str(v)[:200])
  else: print(str(x)[:1000])
 except Exception as e: print('ERR',e)
PY"
```
<details><summary>output</summary>

```

 /pipeline/foundationpose/episode_000000.pkl
<class 'dict'>
frames <class 'list'> 778
camera_id <class 'int'> 0
fp_poses_cam (778, 1, 4, 4) float32
fp_candidate_count (778,) int32
fp_selected_candidate_index (778,) int32
fp_selected_iou (778,) float32
fp_selected_visibility (778,) float32
fp_retry_count (778,) int32
fp_reliable (778,) bool
fp_oracle_rotation_error_deg (778,) float32
fp_translation_threshold_m (778,) float32
fp_selected_translation_distance_m (778,) float32
gt_object_visibility_ratio (778,) float32
gt_object_visibility_valid (778,) bool
gt_object_rendered_pixels (778,) int32
gt_object_visible_pixels (778,) int32
obj_pose_world (778, 4, 4) float32
obj_rot_init (778, 3, 3) float32
obj_t_init (778, 3) float32
metadata dict ['source', 'depth_root', 'depth_source', 'depth_input …[+2202 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"T_world_mesh\\|depth correction\\|pose_abs\\|aligned\" -n /code/reconstruction/modules/v2d_cari4d /code/reconstruction/modules/v2d_hoi_object_reconstruction /code/reconstruction/modules/v2d_pipelines 2>/dev/null | head -200"
```
<details><summary>output</summary>

```
_reconstruction/tools/evaluate_mesh_quality.py:453:        raise ValueError("aligned mesh bounds are zero or invalid")
/code/reconstruction/modules/v2d_hoi_object_reconstruction/tools/evaluate_mesh_quality.py:538:    aligned_bounds = aligned_bounding_box_metrics(
/code/reconstruction/modules/v2d_hoi_object_reconstruction/tools/evaluate_mesh_quality.py:580:        "aligned_bounding_box": aligned_bounds,
/code/reconstruction/modules/v2d_hoi_object_reconstruction/tools/evaluate_mesh_quality.py:632:            "Reference axis-aligned bounding-box diagonal; reported as percent."
/code/reconstruction/modules/v2d_hoi_object_reconstruction/tools/evaluate_mesh_quality.py:641:        "aligned_bounding_box": (

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '600,760p' /code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py"
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
                    profiler.record("host_to_ …[+8578 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '500,900p' /code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py"
```
<details><summary>output</summary>

```
        cfg: MHRPostOptConfig | Mapping[str, Any] | None = None,
        *,
        object_faces: Any | None = None,
        mhr_layer: Any | None = None,
        sdf_fn: Callable[[Any], Any] | None = None,
        hand_surface_vertex_indices: Any | None = None,
        _contact_activation_state: Mapping[str, Any] | None = None,
    ) -> None:
        if cfg is None:
            cfg = MHRPostOptConfig()
        elif isinstance(cfg, Mapping):
            cfg = MHRPostOptConfig(**cfg)
        self.cfg = cfg.checked()
        torch = _torch()
        device_name = self.cfg.device
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.dtype = torch.float32

        total = le …[+26831 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 print('\\n',p)
 try:
  x=torch.load(p,map_location='cpu',weights_only=False)
  print(type(x))
  if isinstance(x,dict):
   for k,v in x.items():
    print(k, type(v), getattr(v,'shape',len(v) if hasattr(v,'__len__') else ''))
    if isinstance(v,dict):
     print(' subkeys',list(v)[:100])
    elif hasattr(v,'shape'):
     print(' ',v.dtype, v.flatten()[:5])
    elif isinstance(v,(list,tuple)): print(' sample',str(v[:3])[:500])
  else: print(str(x)[:2000])
 except Exception as e: print('ERR',repr(e))
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
<class 'dict'>
schema <class 'str'> 28
frames <class 'list'> 778
 sample ['000000', '000001', '000002']
frame_meta <class 'list'> 778
 sample [{'frame': '000000', 'src_frame': 0, 'kid': 0}, {'frame': '000001', 'src_frame': 1, 'kid': 0}, {'frame': '000002', 'src_frame': 2, 'kid': 0}]
kid <class 'int'> 
checkpoint <class 'dict'> 2
 subkeys ['path', 'step']
mhr_inference <class 'dict'> 8
 subkeys ['schema', 'checkpoint', 'wandbRun', 'wandbConfigSha256', 'supervisionContract', 'configSource', 'xyzAnchorContract', 'spatialNormalizationContract']
config <class 'dict'> 291
 subkeys ['rgb_files', 'render_files', 'split_file', 'val_split_file', 'render_root', 'rgb_root', 'packed_root', 'packed_format', 'expected_train_sequence_count', 'expected_val_sequence_count', …[+10151 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, numpy as np, pickle
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 print('attrs',dict(f.attrs))
 def walk(name,obj):
  if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype, 'attrs',dict(obj.attrs))
  else: print('GROUP',name,dict(obj.attrs))
 f.visititems(walk)
PY"
```
<details><summary>output</summary>

```
attrs {'complete': True, 'depth_alignment_input_identity_json': '{"depth_backend":"moge2","depth_model_id":"Ruicheng/moge-2-vitl-normal","depth_model_revision":"b135031bae30b5ac2ae141a0e68717795ce38340","depth_source_commit":"925b8ed835a7a9cdb7578ba15c658a0afc969030","direct_mhr":{"mtime_ns":1791141435000000000,"path":"/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/mhr_init/episode_000000.pkl","size":3472357},"export":{"mtime_ns":1791141361000000000,"path":"/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/export/episode_000000/wild_export.json","size":2778},"raw_depth":{"mtime_ns":1791141302000000000,"path":"/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/depth/moge2/raw.npy","size":2753298560},"raw_depth_report":{"mtime_ns":1791141304000000000,"path":"/l/users/han.zhou/v2d/out …[+717 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import torch, numpy as np
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 x=torch.load(p,map_location='cpu',weights_only=False)
 print('\\n',p)
 print('metadata',x.get('metadata'))
 print('pr pose_abs',x['pr']['pose_abs'].shape,x['pr']['pose_abs'].dtype)
 print('initial',x['pr_initial']['pose_abs'].shape)
 print('in pose',x['in']['pose_abs'].shape)
 if 'postopt' in x:
  po=x['postopt']
  print('postopt keys',po.keys())
  print('mode',po.get('mode'),'frame_indices',po.get('frame_indices')[:20],po.get('frame_indices').shape)
  print('optimized',po.get('optimized_parameters'))
  print('final diag',po.get('final_diagnostics'))
  print('config',po.get('config'))
 print('pr pose samples',x['pr']['pose_abs'][[0,100,300,500,700,777]])
 print('obs keys shape')
 for k,v in x['observations'].items():
  if hasattr(v,'shape'): print(k,v.shape,v.dtype)
  else: print(k,type(v),str(v)[:100])
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
metadata {'ground_truth_used': False, 'window_length': 96, 'window_stride': 96, 'overlap_policy': 'first_occurrence', 'depth_source': '/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/depth/moge2/aligned.h5', 'depth_backend': 'moge2', 'depth_alignment_method': 'huber-log-ratio-to-direct-mhr-depth-v1', 'mhr_init_source': '/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/mhr_init/episode_000000.pkl', 'foundationpose_source': '/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/foundationpose/episode_000000.pkl', 'object_mesh': '/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/export/episode_000000/object_mesh/output_aligned.glb', 'mhr_xyz_anchor_contract': {'schema': 'cari4d.mhr_xyz_anchor_contract.v1', 'anchorType': 'root_joint_1', 'rootJointIndex' …[+5431 chars]
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
import h5py, numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for g in ['aligned','raw']:
  d=f[g]['front_stereo_camera_left']
  print(g,d.shape,d.dtype)
  for i in [0,1,777]:
   v=d[i]
   print(i,type(v),getattr(v,'shape',None),getattr(v,'dtype',None),str(v)[:200])
  if d.dtype==object:
   try:
    print('read',d[0],type(d[0]))
   except Exception as e: print(e)
 print('frame names',f['frame_names/front_stereo_camera_left'][:5])
 print('scale',f['alignment/front_stereo_camera_left/scale'][:5])
PY"
```
<details><summary>output</summary>

```
aligned (778,) object
0 <class 'numpy.ndarray'> (659465,) uint8 [137  80  78 ...  66  96 130]
1 <class 'numpy.ndarray'> (655843,) uint8 [137  80  78 ...  66  96 130]
777 <class 'numpy.ndarray'> (634420,) uint8 [137  80  78 ...  66  96 130]
read [137  80  78 ...  66  96 130] <class 'numpy.ndarray'>
raw (778,) object
0 <class 'numpy.ndarray'> (605192,) uint8 [137  80  78 ...  66  96 130]
1 <class 'numpy.ndarray'> (604821,) uint8 [137  80  78 ...  66  96 130]
777 <class 'numpy.ndarray'> (590181,) uint8 [137  80  78 ...  66  96 130]
read [137  80  78 ...  66  96 130] <class 'numpy.ndarray'>
frame names [b'000000' b'000001' b'000002' b'000003' b'000004']
scale [1.118034  1.107011  1.1068578 1.0982063 1.1228015]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch, numpy as np
p='/pipeline/inference/refined.pth'; x=torch.load(p,map_location='cpu',weights_only=False)
po=x['postopt']
for k,v in po.items():
 print('\\nKEY',k,type(v))
 if isinstance(v,dict):
  for kk,vv in v.items(): print(' ',kk,type(vv),getattr(vv,'shape',None),str(vv)[:500])
 elif hasattr(v,'shape'): print(v.shape,v.dtype,str(v.flatten()[:20]))
 else: print(str(v)[:2000])
print('pr pose abs postopt',x['pr'].get('pose_abs_postopt').shape)
print('pr pose diffs postopt vs init',np.linalg.norm(x['pr']['pose_abs_postopt'][...,3]-x['pr']['pose_abs'][...,3],axis=-1))
PY"
```
<details><summary>output</summary>

```

KEY mode <class 'str'>
smplh_parity

KEY checkpoint_schema <class 'str'>
cari4d.mhr_postopt_smplh_parity.v10

KEY batch_sampling <class 'str'>
full_clip_v1

KEY resolved_batch_size <class 'int'>
778

KEY object_mesh_loader_revision <class 'str'>
trimesh-scene-node-transforms-v1

KEY frame_indices <class 'list'>
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,  …[+10625 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for i in [0,300,777]:
  for g in ['raw','aligned']:
   b=f[g]['front_stereo_camera_left'][i]
   im=cv2.imdecode(b,cv2.IMREAD_UNCHANGED)
   print(i,g,im.shape,im.dtype,im.min(),im.max(),np.percentile(im,[1,25,50,75,99]),'scale',f['alignment/front_stereo_camera_left/scale'][i])
raw=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r')
print('raw npy',raw.shape,raw.dtype,raw[0].min(),raw[0].max(),np.percentile(raw[0],[1,25,50,75,99]))
PY"
```
<details><summary>output</summary>

```
0 raw (1152, 1536) uint16 1579 8808 [1652. 2542. 3599. 4769. 8205.] scale 1.118034
0 aligned (1152, 1536) uint16 1765 9847 [1846. 2842. 4023. 5331. 9173.] scale 1.118034
300 raw (1152, 1536) uint16 1594 8732 [1672. 2451. 3574. 4747. 8158.] scale 1.104547
300 aligned (1152, 1536) uint16 1760 9644 [1846. 2707. 3947. 5243. 9010.] scale 1.104547
777 raw (1152, 1536) uint16 1603 9080 [1681. 2596. 3653. 4798. 8054.] scale 1.0894964
777 aligned (1152, 1536) uint16 1746 9892 [1831. 2828. 3979. 5227. 8774.] scale 1.0894964
raw npy (778, 1152, 1536) uint16 1579 8808 [1652. 2542. 3599. 4769. 8205.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=m.to_geometry()
print('visual',vars(g.visual).keys())
for k,v in vars(g.visual).items():
 print(k,type(v),getattr(v,'shape',None),str(v)[:300])
if hasattr(g.visual,'material'):
 print('material',vars(g.visual.material))
print('uv',g.visual.uv.shape,g.visual.uv.min(0),g.visual.uv.max(0))
print('transform',m.graph.nodes_geometry)
PY"
```
<details><summary>output</summary>

```
visual dict_keys(['vertex_attributes', '_cache', 'material', 'face_materials', 'mesh'])
vertex_attributes <class 'trimesh.caching.DataStore'> None <trimesh.caching.DataStore object at 0x76367f344090>
_cache <class 'trimesh.caching.Cache'> None <trimesh.caching.Cache object at 0x76367f344150>
material <class 'trimesh.visual.material.PBRMaterial'> None <trimesh.visual.material.PBRMaterial object at 0x76367f344250>
face_materials <class 'NoneType'> None None
mesh <class 'trimesh.base.Trimesh'> None <trimesh.Trimesh(vertices.shape=(51247, 3), faces.shape=(95759, 3))>
material {'_data': {'baseColorFactor': array([255, 255, 255, 255], dtype=uint8), 'metallicFactor': 0.0, 'roughnessFactor': 0.5, 'baseColorTexture': <PIL.Image.Image image mode=RGB size=5007x4510 at 0x76372E9FABD0>, 'doubleSided':  …[+103 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, trimesh, json
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=np.asarray(g.vertices); n,d=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(n); d=float(d)
sd=(np.einsum('fj,nj->fn',v,n)+d) # F,V
mins=sd.min(1); maxsd=sd.max(1)
np.set_printoptions(precision=3,suppress=True,linewidth=300)
for i in range(0,778,20): print(i,mins[i],maxsd[i])
print('stats phases')
for a,b in [(0,300),(0,280),(300,620),(620,778),(0,778)]:
 print(a,b,'min sd median/mean/std/min/max',np.median(mins[a:b]),mins[a:b].mean(),mins[a:b].std(),mins[a:b].min(),mins[a:b].max())
# align vertical to floor for static 0:290, 630:778 -> translation delta mean -min
for a,b in [(0,290),(280,320),(620,778)]:
 dm=-mins[a:b]
 print('seg',a,b,'delta y to floor mean med std',dm.mean(),np.median(dm),dm.std(), 'world candidate',T[a:b,:3,3]+np.stack([dm*0,dm,dm*0],1))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, trimesh, json
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=np.asarray(g.vertices); plane=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(plane[:3],float); d=float(plane[3])
# homogeneous T
vh=np.c_[v,np.ones(len(v))]
w=np.einsum('fij,vj->fvi',T,vh)[:,:,:3]
sd=np.einsum('fvi,i->fv',w,n)+d
mins=sd.min(1); maxsd=sd.max(1)
np.set_printoptions(precision=3,suppress=True,linewidth=300)
for i in range(0,778,20): print(i,mins[i],maxsd[i])
print('stats phases')
for a,b in [(0,300),(0,280),(300,620),(620,778),(0,778)]:
 print(a,b,'min sd median/mean/std/min/max',np.median(mins[a:b]),mins[a:b].mean(),mins[a:b].std(),mins[a:b].min(),mins[a:b].max())
for a,b in [(0,290),(280,320),(620,778)]:
 dm=-mins[a:b]
 print('seg',a,b,'delta along n mean med std',dm.mean(),np.median(dm),dm.std())
PY"
```
<details><summary>output</summary>

```
0 -0.02692461007049407 0.47218639166403586
20 -0.021845270625510116 0.47110904562478906
40 -0.02669484316452686 0.46727871294472323
60 -0.033066158710213234 0.4630488251561695
80 -0.014825616137334796 0.47768915342502993
100 -0.03958202529648425 0.45256221773179695
120 -0.04315109967926967 0.44882834797604443
140 -0.05712247956288863 0.4347217065309359
160 -0.05211768778801762 0.4400834045471781
180 -0.045000330439391334 0.44128157831190273
200 -0.02206118384507172 0.4633327403459201
220 -0.023048264023567633 0.4676258140818451
240 -0.02253576252667866 0.4564484648030449
260 -0.020638289129456755 0.4605060576202351
280 -0.021413306566115065 0.472025446839599
300 -0.0349321871406687 0.5644532286320674
320 -0.004814460708355117 0.6716064389281976
340 -0.013395316734355456 0.7252340448402668
 …[+1974 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p /work/scratch/contact && python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
segments=[range(240,361,5),range(570,671,5)]
with h5py.File('/work/inputs/masks.h5','r') as f: om=f['object']; pm=f['person']
for si,idxs in enumerate(segments):
 frames=[]
 for i in idxs:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); assert ok
  ov=im.copy(); o=om[i].astype(bool);p=pm[i].astype(bool)
  ov[o]=(0.3*ov[o]+0.7*np.array([0,255,0])).astype(np.uint8)
  ov[p]=(0.7*ov[p]+0.3*np.array([255,0,0])).astype(np.uint8)
  cv2.rectangle(ov,(0,0),(110,40),(0,0,0),-1); cv2.putText(ov,str(i),(5,32),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
  frames.append(cv2.resize(ov,(512,384)))
 rows=[np.concatenate(frames[j:j+5],1) for j in range(0,len(frames),5)]
 cv2.imwrite(f'/work/scratch/contact/trans{si}.jpg',np.concatenate(rows,0))
 cap.release()
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
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

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; c=z['contact_prob_right_left']; k=z['human_keypoints_world']
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:]
ys,xs=np.where(o); 
# compute per frame sequential mask IoU and centroid
area=o.sum((1,2)); cent=np.full((778,2),np.nan); iou=np.ones(778)
for i in range(778):
 yy,xx=np.where(o[i]); 
 if len(xx): cent[i]=[xx.mean(),yy.mean()]
 if i:
  inter=np.logical_and(o[i],o[i-1]).sum(); uni=np.logical_or(o[i],o[i-1]).sum(); iou[i]=inter/uni
speed=np.r_[0,np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)]
rs=np.r_[0,R.from_matrix(np.einsum('nji,njk->nik',T[:-1,:3,:3],T[1:,:3,:3])).magnitude()*180/np.pi]
cs=np.r_[0,np.linalg.norm(cent[1:]-cent[:-1],axis=1)]
np.set_printoptions(precision=3,suppress=True,linewidth=300)
print('frame speed rotspeed contactL R maskiou dcent area')
for i in range(250,351): print(i,f'{speed[i]:.4f}',f'{rs[i]:.3f}',f'{c[i,0]:.3f}',f'{c[i,1]:.3f}',f'{iou[i]:.3f}',f'{cs[i]:.2f}',area[i])
print('--- final transition ---')
for i in range(580,661): print(i,f'{speed[i]:.4f}',f'{rs[i]:.3f}',f'{c[i,0]:.3f}',f'{c[i,1]:.3f}',f'{iou[i]:.3f}',f'{cs[i]:.2f}',area[i])
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
import numpy as np,h5py
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; c=z['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:]
area=o.sum((1,2)); cent=np.full((778,2),np.nan); iou=np.ones(778)
for i in range(778):
 yy,xx=np.where(o[i])
 if len(xx): cent[i]=[xx.mean(),yy.mean()]
 if i:
  inter=np.logical_and(o[i],o[i-1]).sum(); uni=np.logical_or(o[i],o[i-1]).sum(); iou[i]=inter/uni
speed=np.r_[0,np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)]
rs=np.r_[0,R.from_matrix(np.einsum('nji,njk->nik',T[:-1,:3,:3],T[1:,:3,:3])).magnitude()*180/np.pi]
cs=np.r_[0,np.linalg.norm(cent[1:]-cent[:-1],axis=1)]
print('frame speed rotspeed contactL R maskiou dcent area')
for i in range(250,351): print(i,f'{speed[i]:.4f}',f'{rs[i]:.3f}',f'{c[i,0]:.3f}',f'{c[i,1]:.3f}',f'{iou[i]:.3f}',f'{cs[i]:.2f}',area[i])
print('--- final transition ---')
for i in range(580,661): print(i,f'{speed[i]:.4f}',f'{rs[i]:.3f}',f'{c[i,0]:.3f}',f'{c[i,1]:.3f}',f'{iou[i]:.3f}',f'{cs[i]:.2f}',area[i])
PY"
```
<details><summary>output</summary>

```
frame speed rotspeed contactL R maskiou dcent area
250 0.0045 0.510 1.000 1.000 0.960 1.17 19489
251 0.0038 0.439 1.000 1.000 0.964 1.57 18873
252 0.0053 0.735 1.000 1.000 0.965 1.38 19476
253 0.0038 0.391 1.000 1.000 0.965 1.25 18862
254 0.0016 0.274 1.000 1.000 0.963 1.50 19509
255 0.0036 0.434 1.000 1.000 0.964 1.52 18886
256 0.0074 0.476 1.000 1.000 0.965 1.37 19505
257 0.0026 0.362 1.000 1.000 0.993 0.14 19447
258 0.0106 0.997 1.000 1.000 0.964 1.33 18841
259 0.0044 0.524 1.000 1.000 0.965 1.35 19404
260 0.0109 0.505 1.000 1.000 0.985 0.21 19492
261 0.0053 0.194 1.000 1.000 0.966 1.45 18864
262 0.0020 0.468 1.000 1.000 0.965 1.42 19502
263 0.0020 0.103 1.000 1.000 0.992 0.24 19507
264 0.0020 0.201 1.000 1.000 0.994 0.17 19483
265 0.0031 0.250 1.000 1.000 0.993 0.24 19478
266 0.0020 0. …[+7648 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
segments=[range(240,361,5),range(570,671,5)]
with h5py.File('/work/inputs/masks.h5','r') as f: om=f['object']; pm=f['person']
 for si,idxs in enumerate(segments):
  frames=[]
  for i in idxs:
   cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); assert ok
   ov=im.copy(); o=om[i].astype(bool);p=pm[i].astype(bool)
   ov[o]=(0.3*ov[o]+0.7*np.array([0,255,0])).astype(np.uint8)
   ov[p]=(0.7*ov[p]+0.3*np.array([255,0,0])).astype(np.uint8)
   cv2.rectangle(ov,(0,0),(110,40),(0,0,0),-1); cv2.putText(ov,str(i),(5,32),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
   frames.append(cv2.resize(ov,(512,384)))
  rows=[np.concatenate(frames[j:j+5],1) for j in range(0,len(frames),5)]
  cv2.imwrite(f'/work/scratch/contact/trans{si}.jpg',np.concatenate(rows,0))
cap.release()
print('done')
PY"
```
<details><summary>output</summary>

```
  File "<stdin>", line 5
    for si,idxs in enumerate(segments):
IndentationError: unexpected indent

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
segments=[range(240,361,5),range(570,671,5)]
with h5py.File('/work/inputs/masks.h5','r') as f:
    om=f['object']; pm=f['person']
    for si,idxs in enumerate(segments):
        frames=[]
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); assert ok
            ov=im.copy(); o=om[i].astype(bool);p=pm[i].astype(bool)
            ov[o]=(0.3*ov[o]+0.7*np.array([0,255,0])).astype(np.uint8)
            ov[p]=(0.7*ov[p]+0.3*np.array([255,0,0])).astype(np.uint8)
            cv2.rectangle(ov,(0,0),(110,40),(0,0,0),-1); cv2.putText(ov,str(i),(5,32),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
            frames.append(cv2.resize(ov,(512,384)))
        rows=[np.concatenate(frames[j:j+5],1) for j in range(0,len(frames),5)]
        cv2.imwrite(f'/work/scratch/contact/trans{si}.jpg',np.concatenate(rows,0))
cap.release()
print('done')
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 16, in <module>
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 2560 and the array at index 4 has size 512

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
segments=[list(range(240,361,5)),list(range(570,671,5))]
with h5py.File('/work/inputs/masks.h5','r') as f:
    om=f['object']; pm=f['person']
    for si,idxs in enumerate(segments):
        frames=[]
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); assert ok
            ov=im.copy(); o=om[i].astype(bool);p=pm[i].astype(bool)
            ov[o]=(0.3*ov[o]+0.7*np.array([0,255,0])).astype(np.uint8)
            ov[p]=(0.7*ov[p]+0.3*np.array([255,0,0])).astype(np.uint8)
            cv2.rectangle(ov,(0,0),(110,40),(0,0,0),-1); cv2.putText(ov,str(i),(5,32),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
            frames.append(cv2.resize(ov,(512,384)))
        while len(frames)%5: frames.append(np.zeros_like(frames[0]))
        rows=[np.concatenate(frames[j:j+5],1) for j in range(0,len(frames),5)]
        cv2.imwrite(f'/work/scratch/contact/trans{si}.jpg',np.concatenate(rows,0))
cap.release()
print('done')
PY"
```
<details><summary>output</summary>

```
done

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_mask_metrics.py <<'PY'
import numpy as np, cv2, trimesh, json, h5py, time
from scipy.spatial.transform import Rotation as R
inp='/work/inputs'; z=np.load(inp+'/reconstruction.npz'); T=z['T_world_mesh']
cam=json.load(open(inp+'/camera.json')); K=np.array(cam['K'],float); Twc=np.array(cam['T_world_cam'],float); Tcw=np.linalg.inv(Twc)
g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry(); v=np.asarray(g.vertices,float); f=np.asarray(g.faces)
# subsample scale
S=0.5; W=int(1536*S); H=int(1152*S); Ks=K.copy(); Ks[:2]*=S
with h5py.File(inp+'/masks.h5','r') as h:
    om=h['object'][:]; pm=h['person'][:]
ious=[]; prec=[]; rec=[]; areas=[]; ras=[]
for i in range(len(T)):
    vc=np.c_[v,np.ones(len(v))]@T[i].T
    vc=(Tcw@vc.T).T[:,:3]
    uv=(K@vc.T).T; uv=uv[:,:2]/uv[:,2:3]
    uv*=S
    mask=np.zeros((H,W),np.uint8)
    pts=np.round(uv[f]).astype(np.int32)
    cv2.fillPoly(mask,pts,1)
    pred=mask.astype(bool); obs=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool); per=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
    valid=~per
    inter=(pred&obs&valid).sum(); union=((pred|obs)&valid).sum()
    ious.append(inter/union if union else 1); prec.append(inter/max((pred&valid).sum(),1)); rec.append(inter/max((obs&valid).sum(),1)); areas.append(obs.sum()); ras.append(pred.sum())
np.savez('/work/scratch/baseline_mask_metrics.npz',iou=ious,prec …[+394 chars]
```
<details><summary>output</summary>

```
0 290 iou 0.7396970947643525 0.7170384830112528 prec 0.8922132765473533 rec 0.8107494624098519
290 650 iou 0.8178518905584469 0.8218806416593503 prec 0.938282303255194 rec 0.8639370442928731
650 778 iou 0.8093222435221193 0.8093193821343725 prec 0.9416070163634036 rec 0.852156847207624
samples
0 0.8138075313807531 0.9602962353425221 0.8421432437308317 5543 4862
20 0.8136547598099595 0.9465711361310133 0.8528218369605312 5422 4886
40 0.8056524773203071 0.9412963717896453 0.8482733284349743 5444 4906
60 0.8012900913814729 0.9192189105858171 0.8619892058596762 5188 4865
80 0.777103209019948 0.8986960882647944 0.8517110266159695 5260 4985
100 0.7467271775178914 0.9021509911429777 0.8125356125356126 5265 4742
120 0.7632371170533027 0.9135226791013141 0.8226760832219889 5239 4718
140 0.668081494 …[+2247 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
for a,b in [(0,290),(0,300),(290,650),(650,778),(0,778)]:
 rr=R.from_matrix(T[a:b,:3,:3]); m=rr.mean(); err=(rr*m.inv()).magnitude()*180/np.pi
 tt=T[a:b,:3,3]; print(a,b,'rot mean err mean med max',err.mean(),np.median(err),err.max(),'t mean',tt.mean(0),'t std',tt.std(0),'t max dev',np.linalg.norm(tt-tt.mean(0),axis=1).max())
 # orientation matrix pair differences
 print('  mean rot\\n',m.as_matrix())
PY"
```
<details><summary>output</summary>

```
0 290 rot mean err mean med max 0.6448207711462477 0.5525165237768122 3.75966551920913 t mean [-0.11155586  0.9901906   2.7508483 ] t std [0.00816838 0.00999813 0.02891893] t max dev 0.07044445
  mean rot
 [[-0.03326355 -0.61304286  0.78934909]
 [ 0.99686561 -0.07707355 -0.01785028]
 [ 0.07178093  0.78628119  0.61368508]]
0 300 rot mean err mean med max 0.8554558809070834 0.5953176504847637 7.838421440407468 t mean [-0.11244304  0.98876405  2.7508566 ] t std [0.00935614 0.01269708 0.02848193] t max dev 0.07098856
  mean rot
 [[-0.03530811 -0.61347388  0.7889253 ]
 [ 0.99699126 -0.07613052 -0.01457959]
 [ 0.0690055   0.78603686  0.61431612]]
290 650 rot mean err mean med max 21.956014456801334 21.153300748028563 44.1514304330385 t mean [0.12593938 0.72407216 2.937267  ] t std [0.18989675 0. …[+817 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_nvdiff.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py
import nvdiffrast.torch as dr
inp='/work/inputs'; z=np.load(inp+'/reconstruction.npz'); T=z['T_world_mesh'][0].astype(np.float32)
cam=json.load(open(inp+'/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Tcw=np.linalg.inv(Twc).astype(np.float32)
g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry(); v=np.asarray(g.vertices,np.float32); f=np.asarray(g.faces,np.int32)
H,W=576,768; sx=W/1536; sy=H/1152; Ks=K.copy(); Ks[0]*=sx; Ks[1]*=sy
Tc=np.eye(4,dtype=np.float32); Tc[:3,:3]=Tcw[:3,:3]; Tc[:3,3]=Tcw[:3,3]
# mesh world then cam
vh=np.c_[v,np.ones(len(v),np.float32)]; w=(vh@T.T).astype(np.float32); c=(np.c_[w[:,:3],np.ones(len(w),np.float32)]@Tc.T)[:,:3]
fx,fy=Ks[0,0],Ks[1,1]; cx,cy=Ks[0,2],Ks[1,2]
x=c[:,0];y=c[:,1];zz=c[:,2]
near,far=.1,100.
clip=np.stack([2*fx/W*x+(2*cx/W-1)*zz, -(2*fy/H*y+(2*cy/H-1)*zz), (far+near)/(far-near)*zz-2*far*near/(far-near), zz],1)
glctx=dr.RasterizeCudaContext()
rast,hard=dr.rasterize(glctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),resolution=[H,W])
mask=(rast[0,:,:,3]>0).cpu().numpy()
cv2.imwrite('/work/scratch/test_rast.png',(mask*255).astype(np.uint8))
# compare cv2
uv=(Ks@c.T).T;uv=uv[:,:2]/uv[:,2:3]
m2=np.zeros((H,W),np.uint8);cv2.fillPoly(m2,np.round(uv[f]).astype(np.int32),1)
print('iou masks',np.logical_and(mask,m2).sum()/np.logical_or(mas …[+1136 chars]
```
<details><summary>output</summary>

```
iou masks 0.0 4417 4862
zr torch.Size([1, 576, 768, 1]) 0.0 0.9397072792053223
iou obs 0.0 4417 5543

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/test_nvdiff.py'
s=open(p).read().replace(\"-(2*fy/H*y+(2*cy/H-1)*zz)\",\"(2*fy/H*y+(2*cy/H-1)*zz)\")
open(p,'w').write(s)
PY
python /work/scratch/test_nvdiff.py"
```
<details><summary>output</summary>

```
iou masks 0.9012497439049375 4418 4862
zr torch.Size([1, 576, 768, 1]) 0.0 0.9397072792053223
iou obs 0.7797034125424335 4418 5543

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/test_texture.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py
import nvdiffrast.torch as dr
inp='/work/inputs'; T=np.load(inp+'/reconstruction.npz')['T_world_mesh'][0].astype(np.float32)
cam=json.load(open(inp+'/camera.json')); K=np.array(cam['K'],np.float32); Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry(); v=np.asarray(g.vertices,np.float32); f=np.asarray(g.faces,np.int32); uv=np.asarray(g.visual.uv,np.float32); tex=np.asarray(g.visual.material.baseColorTexture.convert('RGB'),np.float32)/255
print('tex',tex.shape,uv.min(0),uv.max(0))
H,W=576,768; sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy
vh=np.c_[v,np.ones(len(v),np.float32)];w=vh@T.T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3]
fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];x,y,zz=c[:,0],c[:,1],c[:,2];near,far=.1,100.
clip=np.stack([2*fx/W*x+(2*cx/W-1)*zz,2*fy/H*y+(2*cy/H-1)*zz,(far+near)/(far-near)*zz-2*far*near/(far-near),zz],1)
ctx=dr.RasterizeCudaContext();rast,_=dr.rasterize(ctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),resolution=[H,W])
uvr,_=dr.interpolate(torch.from_numpy(uv)[None].cuda(),rast,torch.from_numpy(f).cuda());print('uvr',uvr.shape,uvr.min().item(),uvr.max().item())
mask=rast[0,:,:,3]>0
# decode image
cap=cv2.VideoCapture(inp+'/video.mp4');ok,im=cap.read();cap.release();im=cv2.resize(im,(W,H));imf=im.asty …[+1031 chars]
```
<details><summary>output</summary>

```
tex (4510, 5007, 3) [0.000699   0.00077599] [0.999301 0.999224]
uvr torch.Size([1, 576, 768, 2]) 0.0 0.9936411380767822
Traceback (most recent call last):
  File "/work/scratch/test_texture.py", line 26, in <module>
    valid=mask&om&~pm
          ~~~~^~~
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/mask=rast\\[0,:,:,3\\]>0/mask=(rast[0,:,:,3]>0).cpu().numpy()/\" /work/scratch/test_texture.py
python /work/scratch/test_texture.py"
```
<details><summary>output</summary>

```
tex (4510, 5007, 3) [0.000699   0.00077599] [0.999301 0.999224]
uvr torch.Size([1, 576, 768, 2]) 0.0 0.9936411380767822
flip False False median 0.21830067038536072 mean 0.22269414365291595 frac<.08 0.08318056911230087
flip False True median 0.2352941334247589 mean 0.24650448560714722 frac<.08 0.008478459902107716
flip True False median 0.21101686358451843 mean 0.22363314032554626 frac<.08 0.10265810787677765
flip True True median 0.21623560786247253 mean 0.23068231344223022 frac<.08 0.06691108644008636

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
im=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().visual.material.baseColorTexture
im.thumbnail((768,768)); im.save('/work/scratch/texture_thumb.jpg')
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
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,280);ok,im=cap.read();cap.release();cv2.imwrite('/work/scratch/frame280.jpg',im)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/overlay_selected.py <<'PY'
import numpy as np, cv2,trimesh,json,h5py
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],float);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices);f=np.asarray(g.faces)
S=.5;W=768;H=576;Ks=K.copy();Ks[:2]*=S
cap=cv2.VideoCapture(inp+'/video.mp4')
with h5py.File(inp+'/masks.h5','r') as h: om=h['object'];pm=h['person']
frames=[]
for i in [0,100,200,280,300,350,400,450,500,550,600,650,700,777]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.resize(im,(W,H));vc=np.c_[v,np.ones(len(v))]@T[i].T;vc=(Tcw@vc.T).T[:,:3];uv=(K@vc.T).T;uv=uv[:,:2]/uv[:,2:3]*S;mask=np.zeros((H,W),np.uint8);cv2.fillPoly(mask,np.round(uv[f]).astype(np.int32),1)
 obs=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST);per=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
 edges=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)
 out=im.copy(); out[edges]=(0,255,0)
 # obs boundary blue
 oe=cv2.morphologyEx(obs,cv2.MORPH_GRADIENT,np.ones((4,4),np.uint8)).astype(bool)&~per.astype(bool); out[oe]=(0,0,255)
 cv2.rectangle(out,(0,0),(100,35),(0,0,0),-1);cv2.putText(out,str(i),(5,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);frames.append(out)
cap.release()
rows=[np.concatenate(fra …[+164 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/overlay_selected.py", line 9, in <module>
    obs=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST);per=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
                   ~~^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 870, in __getitem__
    if self._fast_read_ok and (new_dtype is None):
       ^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/base.py", line 534, in __get__
    value = obj.__dict__[self.func.__name__] = self.func(obj)
                                               ^^^^^ …[+1003 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/overlay_selected.py';s=open(p).read();s=s.replace(\"with h5py.File(inp+'/masks.h5','r') as h: om=h['object'];pm=h['person']\",\"with h5py.File(inp+'/masks.h5','r') as h: om=h['object'][:];pm=h['person'][:]\" );open(p,'w').write(s)
PY
python /work/scratch/overlay_selected.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/overlay_selected.py", line 17, in <module>
    cv2.imwrite('/work/scratch/overlay_selected.jpg',np.concatenate(rows,0))
                                                     ^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 3072 and the array at index 3 has size 1536

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/overlay_selected.py';s=open(p).read();s=s.replace(\"rows=[np.concatenate(frames[j:j+4],1) for j in range(0,len(frames),4)]\",\"while len(frames)%4: frames.append(np.zeros_like(frames[0]))\\nrows=[np.concatenate(frames[j:j+4],1) for j in range(0,len(frames),4)]\");open(p,'w').write(s)
PY
python /work/scratch/overlay_selected.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/depth_metrics.py <<'PY'
import numpy as np,torch,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);f=np.asarray(g.faces,np.int64)
H,W=576,768;sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy;fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];near,far=.1,100.
ctx=dr.RasterizeCudaContext()
with h5py.File(inp+'/masks.h5','r') as h: om=h['object'][:];pm=h['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as h: aligned=h['aligned/front_stereo_camera_left']
res=[]
for i in range(778):
 vh=np.c_[v,np.ones(len(v),np.float32)];w=vh@T[i].astype(np.float32).T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3]
 x,y,z=c[:,0],c[:,1],c[:,2]
 clip=np.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,(far+near)/(far-near)*z-2*far*near/(far-near),z],1)
 rast,_=dr.rasterize(ctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),resolution=[H,W])
 rr=rast[0]; zw,_=dr.interpolate(torch.from_numpy((clip[:,2:3]/clip[:,3:4]))[None].cuda(),rast[None],torch.from_numpy(f).cuda());zw=zw[0,:,:,0].cpu().numpy()
 # invert ndc z to cam z
 ndc=zw; zc=2*far*near/((far+near)-(far-near)*ndc)
 dm=cv2.imdeco …[+940 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_metrics.py", line 13, in <module>
    rast,_=dr.rasterize(ctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),resolution=[H,W])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/l …[+387 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c 'sed -i "s/np.int64)/np.int32)/" /work/scratch/depth_metrics.py
python /work/scratch/depth_metrics.py'
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_metrics.py", line 14, in <module>
    rr=rast[0]; zw,_=dr.interpolate(torch.from_numpy((clip[:,2:3]/clip[:,3:4]))[None].cuda(),rast[None],torch.from_numpy(f).cuda());zw=zw[0,:,:,0].cpu().numpy()
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Fil …[+270 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/,rast\\[None\\]/,rast/\" /work/scratch/depth_metrics.py
python /work/scratch/depth_metrics.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_metrics.py", line 17, in <module>
    dm=cv2.imdecode(aligned[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
                    ~~~~~~~^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 870, in __getitem__
    if self._fast_read_ok and (new_dtype is None):
       ^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/base.py", line 534, in __get__
    value = obj.__dict__[self.func.__name__] = self.func(obj)
                                               ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py …[+926 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/depth_metrics.py';s=open(p).read().replace(\"aligned=h['aligned/front_stereo_camera_left']\",\"aligned=h['aligned/front_stereo_camera_left'][:]\");open(p,'w').write(s)
PY
python /work/scratch/depth_metrics.py"
```
<details><summary>output</summary>

```
0 290 mean 0.13070061415763057 med 0.029587924480438232 p25/75 0.00811965901276161 0.08260001314097437 std 0.14511450570205162
290 650 mean 0.014244050826366421 med -0.05231162905693054 p25/75 -0.08501922534571754 -0.012179385291205512 std 0.0294304221868515
650 778 mean -0.0721015090500714 med -0.11875101923942566 p25/75 -0.1243801899254322 -0.0901573458686471 std -0.06212010560557246
0 [   0.1534    0.0289    0.0754    0.1105    0.1642    0.2474    0.1887 4364.    ]
20 [   0.1796    0.0547    0.0983    0.129     0.1876    0.2902    0.1951 4348.    ]
40 [   0.1878    0.0681    0.1023    0.131     0.1921    0.294     0.2049 4360.    ]
60 [   0.1679    0.0589    0.0933    0.1252    0.1745    0.2504    0.1839 4229.    ]
80 [   0.2582    0.141     0.1735    0.2068    0.2711    0.3525    0.192 …[+2937 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_candidates.py <<'PY'
import numpy as np,cv2,trimesh,json,h5py
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';z=np.load(inp+'/reconstruction.npz');T=z['T_world_mesh'].astype(np.float64)
cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],float);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices);f=np.asarray(g.faces)
plane=json.load(open(inp+'/ground_plane.json'))['plane'];n=np.array(plane[:3]);d=plane[3]
S=.5;W=768;H=576;Ks=K.copy();Ks[:2]*=S
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
o2=np.empty((778,H,W),bool);p2=np.empty((778,H,W),bool)
for i in range(778):
 o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
 p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
def render(TX):
 vc=np.c_[v,np.ones(len(v))]@TX.T;vc=(Tcw@vc.T).T[:,:3];uv=(K@vc.T).T;uv=uv[:,:2]/uv[:,2:3]*S;m=np.zeros((H,W),np.uint8);cv2.fillPoly(m,np.round(uv[f]).astype(np.int32),1);return m.astype(bool)
def mingap(TX):
 w=(np.c_[v,np.ones(len(v))]@TX.T)[:,:3];return (w@n+d).min()
def metrics(m,a,b):
 vals=[];ps=[];rs=[]
 for i in range(a,b):
  valid=~p2[i]; inter=(m&o2[i]&valid).sum();uni=((m|o2[i])&valid).sum(); vals.append(inter/uni)
 return np.mean(vals),np.median(vals),np.min(vals),np.max(vals)
for a,b in [(0,290),(290, …[+816 chars]
```
<details><summary>output</summary>

```

SEG 0 290 baseline metrics (0.0, 0.0, 0.0, 0.0) per-frame mean iou from saved
base saved 0.7396970947643525 0.7170384830112528 0.6673479816044967 0.8465007215007215
mean gap -0.03165643203692148 met (0.7425995866460873, 0.7487362719313291, 0.5588798051835102, 0.7580071174377224) t [-0.11155585  0.99019051  2.75084805] rotrel meanbase 0.0
mean_floor gap 0.0 met (0.5703170671052175, 0.5697385507270961, 0.5520510168907273, 0.5904218533886584) t [-0.11215399  0.95858847  2.74909221] rotrel meanbase 0.0

SEG 290 650 baseline metrics (0.0, 0.0, 0.0, 0.0) per-frame mean iou from saved
base saved 0.8178518905584469 0.8218806416593503 0.7007889166827016 0.884437286088182
mean gap -0.07073933698087576 met (0.3498910107006819, 0.27899345817373267, 0.09395591910929334, 0.7376484765019797) t [0.125939 …[+710 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/sweep_depth_static.py <<'PY'
import numpy as np,cv2,trimesh,json,h5py
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'].astype(np.float64)
cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],float);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices);f=np.asarray(g.faces)
S=.5;W=768;H=576;Ks=K.copy();Ks[:2]*=S
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
o2=np.empty((778,H,W),bool);p2=np.empty((778,H,W),bool)
for i in range(778):
 o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
def render(TX):
 vc=np.c_[v,np.ones(len(v))]@TX.T;vc=(Tcw@vc.T).T[:,:3];uv=(K@vc.T).T;uv=uv[:,:2]/uv[:,2:3]*S;m=np.zeros((H,W),np.uint8);cv2.fillPoly(m,np.round(uv[f]).astype(np.int32),1);return m.astype(bool)
def met(m,a,b):
 vals=[]
 for i in range(a,b):
  q=~p2[i];inter=(m&o2[i]&q).sum();uni=((m|o2[i])&q).sum();vals.append(inter/uni)
 return np.mean(vals),np.median(vals)
for a,b in [(0,290),(650,778)]:
 rr=R.from_matrix(T[a:b,:3,:3]);cm=np.eye(4);cm[:3,:3]=rr.mean().as_matrix();cm[:3,3]=T[a:b,:3,3].mean(0)
 ray=cm[:3,3]-Twc[:3,3];ray/=np.linalg.norm(ray)
 # ray points from cam to object; moving +alpha along ray  …[+238 chars]
```
<details><summary>output</summary>

```

seg 0 290 ray [-0.05828573  0.34090295  0.93828991] base t [-0.11155585  0.99019051  2.75084805]
-0.300 (0.7012275583848188, 0.7084383891081668)
-0.275 (0.7168071833970845, 0.7240630715938723)
-0.250 (0.7273455189171276, 0.7349529613019484)
-0.225 (0.7351128499227066, 0.7432963533291871)
-0.200 (0.7452653103127485, 0.7538336052202284)
-0.175 (0.7485615620159133, 0.757338397685634)
-0.150 (0.7522275486619532, 0.7605131999022798)
-0.125 (0.7567392633416027, 0.7650703400113976)
-0.100 (0.7610673070845395, 0.7697184793895784)
-0.075 (0.7595392343759908, 0.7674798330362659)
-0.050 (0.7530878722979566, 0.7608602082207082)
-0.025 (0.7487296396868798, 0.7563847267690758)
+0.000 (0.7425995866460873, 0.7487362719313291)
+0.025 (0.7336581860757865, 0.7393053737542443)
+0.050 (0.7246895005443862, 0.7 …[+1784 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/frame0_zoom.py <<'PY'
import numpy as np,cv2,trimesh,json,h5py
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'][0];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices);f=np.asarray(g.faces)
cap=cv2.VideoCapture(inp+'/video.mp4');ok,im=cap.read();cap.release()
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][0];pm=h['person'][0]
uv=(K@(Tcw@np.c_[v,np.ones(len(v))]@T.T).T).T;uv=uv[:,:2]/uv[:,2:3];mask=np.zeros(im.shape[:2],np.uint8);cv2.fillPoly(mask,np.round(uv[f]).astype(np.int32),1)
edges=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool); oe=cv2.morphologyEx(om.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)&~pm
out=im.copy();out[edges]=(0,255,0);out[oe]=(0,0,255)
crop=out[680:980,500:920];cv2.imwrite('/work/scratch/frame0_zoom.jpg',crop)
# alpha overlay masks
ov=im.copy();ov[mask.astype(bool)]=(.5*ov[mask.astype(bool)]+.5*np.array([0,255,0])).astype(np.uint8);ov[om]=(.7*ov[om]+.3*np.array([255,0,0])).astype(np.uint8);cv2.imwrite('/work/scratch/frame0_zoom_mask.jpg',ov[680:980,500:920])
print('areas',mask.sum(),om.sum(),'obs/pred',om.sum()/mask.sum())
PY
python /work/scratch/frame0_zoom.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/frame0_zoom.py", line 5, in <module>
    uv=(K@(Tcw@np.c_[v,np.ones(len(v))]@T.T).T).T;uv=uv[:,:2]/uv[:,2:3];mask=np.zeros(im.shape[:2],np.uint8);cv2.fillPoly(mask,np.round(uv[f]).astype(np.int32),1)
           ~~~^~~~~~~~~~~~~~~~~~~~~~~~~
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 51247 is different from 4)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/uv=(K@(Tcw@np.c_\\[v,np.ones(len(v))\\]@T.T).T).T/uv=(K@(Tcw@(np.c_[v,np.ones(len(v))]@T.T).T)[:3]).T/\" /work/scratch/frame0_zoom.py
python /work/scratch/frame0_zoom.py"
```
<details><summary>output</summary>

```
areas 18534 22185 obs/pred 1.1969893169310457

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/texture_eval.py <<'PY'
import numpy as np,torch,trimesh,json,cv2,h5py
import nvdiffrast.torch as dr
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);f=np.asarray(g.faces,np.int32);uv=np.asarray(g.visual.uv,np.float32);tex=np.asarray(g.visual.material.baseColorTexture.convert('RGB'),np.float32)/255
H,W=576,768;sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy;fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];near,far=.1,100.;ctx=dr.RasterizeCudaContext();tex_t=torch.from_numpy(tex).cuda().permute(2,0,1)[None]
cap=cv2.VideoCapture(inp+'/video.mp4')
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
for i in [0,100,200,280,300,350,400,450,500,550,600,650,700,777]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.resize(im,(W,H));imf=im.astype(np.float32)/255
 vh=np.c_[v,np.ones(len(v),np.float32)];w=vh@T[i].astype(np.float32).T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3];x,y,z=c[:,0],c[:,1],c[:,2]
 clip=np.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,(far+near)/(far-near)*z-2*far*near/(far-near),z],1)
 rast,_=dr.rasterize(ctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),resolution=[H,W]) …[+1464 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/texture_eval.py", line 13, in <module>
    X=np.c_[samp[valid].reshape(-1),np.ones(valid.sum())];Y=imf[valid].reshape(-1,3);coef=np.linalg.lstsq(X,Y,rcond=None)[0];pred=np.clip(np.c_[samp.reshape(-1,3),np.ones(samp.shape[0]*samp.shape[1])]@coef,0,1).reshape(H,W,3)
      ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/index_tricks.py", line 418, in __getitem__
    res = self.concatenate(tuple(objs), axis=axis)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 13092 and the array at index 1 has size 4364

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/texture_eval.py';s=open(p).read().replace(\"samp[valid].reshape(-1),np.ones(valid.sum())\",\"samp[valid].reshape(-1,3),np.ones(valid.sum())\").replace(\"np.c_[samp.reshape(-1,3),np.ones(samp.shape[0]*samp.shape[1])]\",\"np.c_[samp.reshape(-1,3),np.ones(H*W)]\");open(p,'w').write(s)
PY
python /work/scratch/texture_eval.py"
```
<details><summary>output</summary>

```
0 mae 0.11310434970771971 med 0.11589709586536581 edgecorr 0.06284035694753884 n 4364
100 mae 0.1143186936687673 med 0.12125763586581859 edgecorr 0.046124738798085846 n 4051
200 mae 0.1166347292613674 med 0.12169581827508674 edgecorr 0.08132495547202025 n 3886
280 mae 0.11307446857793628 med 0.11025366101821182 edgecorr 0.03658757966824696 n 3494
300 mae 0.102071144395637 med 0.09095946487694591 edgecorr 0.024894139809050232 n 3692
350 mae 0.07685425699375115 med 0.05904785033331353 edgecorr 0.04259031054712874 n 4103
400 mae 0.05448806607240396 med 0.03453964653430547 edgecorr 0.09885304486458332 n 4023
450 mae 0.0587294748438938 med 0.04715223571286367 edgecorr 0.14352115900624735 n 4240
500 mae 0.06893722836103551 med 0.06613865606737432 edgecorr 0.09732964135523803 n 4571
550 mae 0.055 …[+428 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
for i in [0,280,600,777]:
 im=cv2.imread(f'/work/scratch/texeval{i}.jpg');
 # image 1728x576, panels each 576
 a=im[:,0:576];b=im[:,576:1152];d=im[:,1152:1728]
 # object approximate same location
 # crop center-bottom
 crop=lambda x:x[300:510,180:650]
 combo=np.concatenate([crop(a),crop(b),crop(d)],1)
 cv2.imwrite(f'/work/scratch/texeval{i}_crop.jpg',combo)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_smoothing.py <<'PY'
import numpy as np, torch, trimesh, json, h5py, cv2
import nvdiffrast.torch as dr
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';zz=np.load(inp+'/reconstruction.npz');base=zz['T_world_mesh'].astype(np.float64);F=len(base)
cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);f=np.asarray(g.faces,np.int32)
H,W=576,768;sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy;fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];near,far=.1,100.;ctx=dr.RasterizeCudaContext()
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
# downsample masks and outer contours
o2=np.empty((F,H,W),bool);p2=np.empty((F,H,W),bool)
for i in range(F):
 o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)

def render(TX):
 vh=np.c_[v,np.ones(len(v),np.float32)];w=vh@TX.astype(np.float32).T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3];x,y,z=c[:,0],c[:,1],c[:,2]
 clip=np.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,(far+near)/(far-near)*z-2*far*near/(far-near),z],1)
 rast,_=dr.rasterize(ctx,torch.from_num …[+3109 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_smoothing.py", line 46, in <module>
    for wl in [5,9,13,17,21]: cands[f'sg{wl}']=smooth_traj(base,wl,2)
                                               ^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/eval_smoothing.py", line 39, in smooth_traj
    vv=R.from_matrix(base[a:b,:3,:3]*C.inv().as_matrix()[None]).as_rotvec()
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "scipy/spatial/transform/_rotation.pyx", line 1143, in scipy.spatial.transform._rotation.Rotation.from_matrix
ValueError: Non-positive determinant (left-handed or null coordinate frame) in rotation matrix 111: [[ 0.24155863 -0.43487754 -0.28523023]
 [-0.53748516  0.00140452  0.39313749]
 [-0.15002437  0.56068029  0.25188961]].

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_smoothing.py';s=open(p).read().replace(\"base[a:b,:3,:3]*C.inv().as_matrix()[None]\",\"np.einsum('ij,njk->nik',C.inv().as_matrix(),base[a:b,:3,:3])\");open(p,'w').write(s)
PY
python /work/scratch/eval_smoothing.py"
```
<details><summary>output</summary>

```
base iou 0.7690533412371425 chamfer 8.837978 pred->obs 6.1389604 obs->pred 2.699018 smoothness rot 0.3968182659453651 trans speed 0.009187855230792841 accel 0.01504989070494929
sg5 iou 0.6318824503919792 chamfer 14.245131 pred->obs 7.904015 obs->pred 6.3411155 smoothness rot 0.3261270325837436 trans speed 0.0030284877695438577 accel 0.0032684433653228477
sg9 iou 0.6318739500223003 chamfer 14.248301 pred->obs 7.904301 obs->pred 6.3439994 smoothness rot 0.30360493538471317 trans speed 0.002149735748158427 accel 0.0015596717691017256
sg13 iou 0.6318809922649817 chamfer 14.242036 pred->obs 7.8988104 obs->pred 6.3432255 smoothness rot 0.2964544870739818 trans speed 0.0018664899308259704 accel 0.0010646466900379667
sg17 iou 0.6318852261169223 chamfer 14.247534 pred->obs 7.903314 obs->pred 6.3442 …[+1321 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/smoothed_candidates.npz');b=z['base'];s=z['sg5']
rel=R.from_matrix(np.einsum('nij,nkj->nik',b[:,:3,:3],s[:,:3,:3])).magnitude()*180/np.pi
print(np.percentile(rel,[0,25,50,75,90,99,100]),rel.argmax())
for i in range(290,651,20):print(i,rel[i],R.from_matrix(s[i,:3,:3]).as_rotvec()*180/np.pi)
PY"
```
<details><summary>output</summary>

```
[3.47761330e-02 4.78380819e-01 1.08293630e+00 2.84025803e+01
 5.29838014e+01 6.38794436e+01 6.42160399e+01] 631
290 63.74514712125066 [-32.57246327  41.97052249  89.59008931]
310 53.63932581164519 [-27.14264258  47.15635697  91.29001838]
330 42.66374345811912 [-21.38301784  53.61468146  92.96549636]
350 39.95841012029353 [-20.05505772  56.14764347  95.2580391 ]
370 31.623716181875817 [-15.17012582  59.66026829  96.0114131 ]
390 27.89654023473627 [-13.10054466  61.18315503  97.11146767]
410 20.29088956217209 [-8.06286464 64.8531021  98.51485026]
430 11.633090072065649 [-2.41724119 67.53549416 99.55502733]
450 9.093398942664361 [-1.0550205  69.12741055 99.57549035]
470 8.583573865555069 [ -1.01341937  70.20201941 100.00298419]
490 5.610725683173729 [  1.0448173   71.73495154 100.95807065]
51 …[+507 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
b=np.load('/work/scratch/smoothed_candidates.npz')['base']
C=R.from_matrix(b[290:650,:3,:3]).mean();
v=R.from_matrix(np.einsum('ij,njk->nik',C.inv().as_matrix(),b[290:650,:3,:3])).as_rotvec()*180/np.pi
vs=savgol_filter(v,5,2,axis=0,mode='interp')
for j,i in enumerate(range(290,650,20)):
 k=i-290
 print(i,'orig',v[k],'smooth',vs[k], 'diff',np.linalg.norm(v[k]-vs[k]))
PY"
```
<details><summary>output</summary>

```
290 orig [ -0.33472851 -44.13436902   1.18077822] smooth [ -0.36027991 -44.1179469    1.15655396] diff 0.03885066991453017
310 orig [ -0.58224621 -37.49850589   0.36425838] smooth [ -0.61118844 -37.41969225   0.37920942] diff 0.08528057225842264
330 orig [ -1.2527929  -29.82376763   0.14725722] smooth [-1.50075545e+00 -2.97376037e+01 -1.67582122e-02] diff 0.3095330477444229
350 orig [ -1.44528645 -26.67438121   1.81676436] smooth [ -1.42753875 -26.68891354   1.68516824] diff 0.13358033941156755
370 orig [ -0.98605847 -21.54988124   0.53887595] smooth [ -1.28456701 -21.59200872   0.64930544] diff 0.321055680832946
390 orig [ -0.95390143 -18.92141369   0.41694992] smooth [ -0.88093809 -19.15357884   0.8789773 ] diff 0.5222007363696397
410 orig [ -0.57851009 -13.79152456   0.75172359] smooth  …[+1364 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_smoothing.py';s=open(p).read().replace(\"(R.from_rotvec(vv)*C).as_matrix()\",\"(C*R.from_rotvec(vv)).as_matrix()\");open(p,'w').write(s)
PY
python /work/scratch/eval_smoothing.py"
```
<details><summary>output</summary>

```
base iou 0.7690533412371425 chamfer 8.837978 pred->obs 6.1389604 obs->pred 2.699018 smoothness rot 0.3968182659453651 trans speed 0.009187855230792841 accel 0.01504989070494929
sg5 iou 0.7712239836367816 chamfer 8.859632 pred->obs 6.160016 obs->pred 2.6996148 smoothness rot 0.16366132587799107 trans speed 0.0030284877695438577 accel 0.0032684433653228477
sg9 iou 0.771313936933367 chamfer 8.85223 pred->obs 6.1625485 obs->pred 2.6896803 smoothness rot 0.1406899962871212 trans speed 0.002149735748158427 accel 0.0015596717691017256
sg13 iou 0.771336645331016 chamfer 8.861986 pred->obs 6.168808 obs->pred 2.693178 smoothness rot 0.1333178705584942 trans speed 0.0018664899308259704 accel 0.0010646466900379667
sg17 iou 0.7713862955280228 chamfer 8.853881 pred->obs 6.162746 obs->pred 2.6911354 smoo …[+1304 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/sweep_depth_chamfer.py <<'PY'
import numpy as np,torch,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);f=np.asarray(g.faces,np.int32)
H,W=576,768;sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy;fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];near,far=.1,100.;ctx=dr.RasterizeCudaContext()
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
o2=np.empty((778,H,W),bool);p2=np.empty((778,H,W),bool)
for i in range(778):o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
def render(TX):
 w=np.c_[v,np.ones(len(v),np.float32)]@TX.astype(np.float32).T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3];x,y,z=c[:,0],c[:,1],c[:,2];clip=np.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,(far+near)/(far-near)*z-2*far*near/(far-near),z],1);rast,_=dr.rasterize(ctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),resolution=[H,W]);return (rast[0,:,:,3]>0).cpu().numpy()
def contour(m):
 …[+1380 chars]
```
<details><summary>output</summary>

```

SEG 0 290
-0.200 iou 0.7614 ch 8.671 p2o 6.421 o2p 2.249
-0.175 iou 0.7632 ch 9.218 p2o 6.666 o2p 2.551
-0.150 iou 0.7641 ch 8.847 p2o 6.690 o2p 2.157
-0.125 iou 0.7639 ch 8.978 p2o 6.776 o2p 2.201
-0.100 iou 0.7613 ch 9.585 p2o 7.010 o2p 2.574
-0.075 iou 0.7576 ch 9.626 p2o 7.065 o2p 2.561
-0.050 iou 0.7464 ch 9.831 p2o 7.226 o2p 2.605
-0.025 iou 0.7393 ch 9.650 p2o 7.066 o2p 2.583
+0.000 iou 0.7313 ch 10.465 p2o 7.565 o2p 2.900
+0.025 iou 0.7205 ch 10.631 p2o 7.692 o2p 2.938
+0.050 iou 0.7088 ch 11.067 p2o 7.875 o2p 3.192
+0.075 iou 0.6947 ch 10.806 p2o 7.577 o2p 3.230
+0.100 iou 0.6815 ch 10.831 p2o 7.530 o2p 3.301

SEG 650 778
-0.200 iou 0.8421 ch 4.498 p2o 2.654 o2p 1.844
-0.175 iou 0.8546 ch 4.184 p2o 2.573 o2p 1.610
-0.150 iou 0.8585 ch 4.228 p2o 2.544 o2p 1.683
-0.125 iou 0.8632 c …[+451 chars]
```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "cat > /work/scratch/sweep_depth_edge.py <<'PY'
import numpy as np,torch,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);f=np.asarray(g.faces,np.int32)
H,W=576,768;sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy;fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];near,far=.1,100.;ctx=dr.RasterizeCudaContext();cap=cv2.VideoCapture(inp+'/video.mp4')
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
o2=np.empty((778,H,W),bool);p2=np.empty((778,H,W),bool);ims=[]
for i in range(778):
 o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();ims.append(cv2.resize(im,(W,H)))
cap.release()
def render(TX):
 w=np.c_[v,np.ones(len(v),np.float32)]@TX.astype(np.float32).T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3];x,y,z=c[:,0],c[:,1],c[:,2];clip=np.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,(far+near)/(far-near)*z-2*far*near/(far-near),z],1);rast,_=dr.raste …[+1223 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/sweep_depth_edge.py", line 10, in <module>
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();ims.append(cv2.resize(im,(W,H)))
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/sweep_depth_edge.py';s=open(p).read()
old=\"\"\"o2=np.empty((778,H,W),bool);p2=np.empty((778,H,W),bool);ims=[]
for i in range(778):
 o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();ims.append(cv2.resize(im,(W,H)))
cap.release()\"\"\"
new=\"\"\"o2=np.empty((778,H,W),bool);p2=np.empty((778,H,W),bool);ims={}
for i in range(778):
 o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
 ok,im=cap.read();assert ok
 if i<290 or i>=650: ims[i]=cv2.resize(im,(W,H))
cap.release()\"\"\"
s=s.replace(old,new)
s=s.replace(\"dts=[];obscs=[]\\nfor i in range(778):\",\"dts={};obscs={}\\nfor i in range(778):\\n if i not in ims: continue\")
open(p,'w').write(s)
PY
python /work/scratch/sweep_depth_edge.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/sweep_depth_edge.py", line 21, in <module>
    gray=cv2.cvtColor(ims[i],cv2.COLOR_BGR2GRAY);ed=cv2.Canny(gray,60,140);dd=(ed==0).astype(np.uint8);dts.append(cv2.distanceTransform(dd,cv2.DIST_L2,5));obscs.append(contour(o2[i]))
                                                                                                       ^^^^^^^^^^
AttributeError: 'dict' object has no attribute 'append'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/sweep_depth_edge.py';s=open(p).read().replace(\"dts.append(cv2.distanceTransform(dd,cv2.DIST_L2,5));obscs.append(contour(o2[i]))\",\"dts[i]=cv2.distanceTransform(dd,cv2.DIST_L2,5);obscs[i]=contour(o2[i])\");open(p,'w').write(s)
PY
python /work/scratch/sweep_depth_edge.py"
```
<details><summary>output</summary>

```

SEG 0 290
-0.250 edge p2o 1.865 med 1.921
-0.225 edge p2o 1.820 med 1.883
-0.200 edge p2o 1.973 med 2.045
-0.175 edge p2o 1.925 med 2.004
-0.150 edge p2o 1.849 med 1.915
-0.125 edge p2o 1.795 med 1.863
-0.100 edge p2o 1.882 med 1.972
-0.075 edge p2o 1.820 med 1.906
-0.050 edge p2o 1.740 med 1.816
-0.025 edge p2o 1.722 med 1.810
-0.000 edge p2o 1.711 med 1.790
+0.025 edge p2o 1.693 med 1.767
+0.050 edge p2o 1.732 med 1.802

SEG 650 778
-0.250 edge p2o 1.098 med 1.107
-0.225 edge p2o 1.012 med 1.018
-0.200 edge p2o 0.940 med 0.936
-0.175 edge p2o 0.857 med 0.855
-0.150 edge p2o 0.849 med 0.843
-0.125 edge p2o 0.877 med 0.852
-0.100 edge p2o 0.925 med 0.890
-0.075 edge p2o 0.970 med 0.932
-0.050 edge p2o 0.983 med 0.940
-0.025 edge p2o 0.986 med 0.951
-0.000 edge p2o 1.050 med 1.005
+0.025 e …[+56 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/final_depth_overlay.py <<'PY'
import numpy as np,cv2,trimesh,json,h5py
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices);f=np.asarray(g.faces)
rr=R.from_matrix(T[650:,:3,:3]);cm=np.eye(4);cm[:3,:3]=rr.mean().as_matrix();cm[:3,3]=T[650:,:3,3].mean(0);ray=cm[:3,3]-Twc[:3,3];ray/=np.linalg.norm(ray)
cap=cv2.VideoCapture(inp+'/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,700);ok,im=cap.read();cap.release()
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][700];pm=h['person'][700]
outs=[]
for al,col in [(.0,(0,255,0)),(-.14,(255,0,255)),(.06,(0,255,255))]:
 cc=cm.copy();cc[:3,3]+=al*ray;vc=(Tcw@(np.c_[v,np.ones(len(v))]@cc.T))[:3].T;uv=(K@vc).T if False else (K@vc.T).T;uv=uv[:,:2]/uv[:,2:3];m=np.zeros(im.shape[:2],np.uint8);cv2.fillPoly(m,np.round(uv[f]).astype(np.int32),1);ed=cv2.morphologyEx(m,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool);out=im.copy();out[ed]=col;outs.append(out)
outs.append(im.copy())
combo=np.concatenate(outs,1);cv2.imwrite('/work/scratch/final_depth_overlay.jpg',combo[560:1080,700:1500])
# closer crop each is full w 1536, object x800-1050 y740-1000
crops=[o[700:1010,760:1080] for o in outs]
cv2.imwrite('/work/scratc …[+104 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/final_depth_overlay.py", line 9, in <module>
    cc=cm.copy();cc[:3,3]+=al*ray;vc=(Tcw@(np.c_[v,np.ones(len(v))]@cc.T))[:3].T;uv=(K@vc).T if False else (K@vc.T).T;uv=uv[:,:2]/uv[:,2:3];m=np.zeros(im.shape[:2],np.uint8);cv2.fillPoly(m,np.round(uv[f]).astype(np.int32),1);ed=cv2.morphologyEx(m,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool);out=im.copy();out[ed]=col;outs.append(out)
                                      ~~~^^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 51247 is different from 4)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/final_depth_overlay.py';s=open(p).read().replace(\"vc=(Tcw@(np.c_[v,np.ones(len(v))]@cc.T))[:3].T\",\"vc=(Tcw@(np.c_[v,np.ones(len(v))]@cc.T))[:3].T\");open(p,'w').write(s)
PY
sed -n '8,10p' /work/scratch/final_depth_overlay.py"
```
<details><summary>output</summary>

```
for al,col in [(.0,(0,255,0)),(-.14,(255,0,255)),(.06,(0,255,255))]:
 cc=cm.copy();cc[:3,3]+=al*ray;vc=(Tcw@(np.c_[v,np.ones(len(v))]@cc.T))[:3].T;uv=(K@vc).T if False else (K@vc.T).T;uv=uv[:,:2]/uv[:,2:3];m=np.zeros(im.shape[:2],np.uint8);cv2.fillPoly(m,np.round(uv[f]).astype(np.int32),1);ed=cv2.morphologyEx(m,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool);out=im.copy();out[ed]=col;outs.append(out)
outs.append(im.copy())

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/final_depth_overlay.py';s=open(p).read().replace(\"(Tcw@(np.c_[v,np.ones(len(v))]@cc.T))[:3].T\",\"(Tcw@(np.c_[v,np.ones(len(v))]@cc.T).T)[:3].T\");open(p,'w').write(s)
PY
python /work/scratch/final_depth_overlay.py"
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
z=np.load('/work/scratch/smoothed_candidates.npz');b=z['base']
for nm in ['sg5','sg9','sg13','sg17','sg21']:
 s=z[nm];rel=R.from_matrix(np.einsum('nji,njk->nik',b[:,:3,:3],s[:,:3,:3])).magnitude()*180/np.pi;dt=np.linalg.norm(s[:,:3,3]-b[:,:3,3],axis=1)
 print(nm,'rot deg mean med p90 max',rel.mean(),np.median(rel),np.percentile(rel,90),rel.max(),'t mean med p90 max',dt.mean(),np.median(dt),np.percentile(dt,90),dt.max())
 for a,bb,n in [(0,290,'init'),(290,650,'move'),(650,778,'final')]:print(' ',n,'r',rel[a:bb].mean(),'t',dt[a:bb].mean(),'max',rel[a:bb].max(),dt[a:bb].max())
PY"
```
<details><summary>output</summary>

```
sg5 rot deg mean med p90 max 0.3940839735011144 0.2986184670788372 0.748090573109306 3.7596655192091615 t mean med p90 max 0.015413176187853875 0.009562393979913309 0.038350783371285724 0.07044471712620667
  init r 0.6448207711462459 t 0.027440297789719705 max 3.7596655192091615 0.07044471712620667
  move r 0.1878315559782927 t 0.004977029500076521 max 0.5339365546555435 0.024338931551092177
  final r 0.40609334061929914 t 0.01751589136800044 max 0.9764154285334012 0.054751285244456986
sg9 rot deg mean med p90 max 0.41653962040551473 0.3374079330590302 0.7576516305758336 3.7596655192091615 t mean med p90 max 0.015924907157331938 0.01032361534098302 0.038350783371285724 0.07044471712620667
  init r 0.6448207711462459 t 0.027440297789719705 max 3.7596655192091615 0.07044471712620667
  move r …[+1661 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/smoothed_candidates.npz');b=z['base'];s=z['sg17']
def rel(A,B):return R.from_matrix(A[:3,:3].T@B[:3,:3]).magnitude()*180/np.pi,np.linalg.norm(B[:3,3]-A[:3,3])
for i in [288,289,290,291,648,649,650,651]:
 print(i,'base prev',rel(b[i-1],b[i]),'sg prev',rel(s[i-1],s[i]))
PY"
```
<details><summary>output</summary>

```
288 base prev (0.7298860380205255, 0.010463954454959654) sg prev (0.0, 0.0)
289 base prev (1.0256692994138388, 0.002644158897044387) sg prev (0.0, 0.0)
290 base prev (0.42830071510664464, 0.005014635735806591) sg prev (3.509824965932196, 0.0369223900750499)
291 base prev (0.6999675989692897, 0.007113761074772599) sg prev (0.6528330197902386, 0.004881972366928126)
648 base prev (0.1680208824559299, 0.010764434252198234) sg prev (0.06510710750635378, 0.0016285995698859738)
649 base prev (0.4553849076992767, 0.022153499247037394) sg prev (0.07111303990320021, 0.0017040552953759749)
650 base prev (0.09435398146478435, 0.020961679095703182) sg prev (0.0898788677034048, 0.009038048649487188)
651 base prev (0.1696231949944822, 0.007379449576616603) sg prev (0.0, 0.0)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];meanR=R.from_matrix(T[:280,:3,:3]).mean();meant=T[:280,:3,3].mean(0)
for i in range(260,341):
 rd=(R.from_matrix(T[i,:3,:3])*meanR.inv()).magnitude()*180/np.pi
 td=np.linalg.norm(T[i,:3,3]-meant)
 if i%2==0 or i>=288: print(i,f'rd={rd:.2f}',f'td={td:.3f}',T[i,:3,3])
PY"
```
<details><summary>output</summary>

```
260 rd=1.18 td=0.009 [-0.1063242  0.9860433  2.7449174]
262 rd=1.46 td=0.005 [-0.10773739  0.9881573   2.7488248 ]
264 rd=1.22 td=0.006 [-0.10585632  0.9886944   2.7499924 ]
266 rd=1.28 td=0.005 [-0.1064233   0.98870915  2.7510254 ]
268 rd=0.93 td=0.007 [-0.10546086  0.9893983   2.7474709 ]
270 rd=1.25 td=0.006 [-0.10638397  0.9880405   2.7522318 ]
272 rd=1.39 td=0.005 [-0.10645178  0.9901085   2.7525399 ]
274 rd=1.19 td=0.007 [-0.10577294  0.9892871   2.7549329 ]
276 rd=1.17 td=0.007 [-0.10651137  0.9880208   2.7553332 ]
278 rd=1.44 td=0.007 [-0.11039456  0.98613846  2.7564538 ]
280 rd=1.96 td=0.011 [-0.11316697  0.9802954   2.7532573 ]
282 rd=1.80 td=0.013 [-0.11647334  0.9786824   2.7524307 ]
284 rd=1.93 td=0.026 [-0.12721252  0.9813451   2.7684565 ]
286 rd=2.37 td=0.027 [-0.13069597  0 …[+3166 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];mr=R.from_matrix(T[680:,:3,:3]).mean();mt=T[680:,:3,3].mean(0)
for i in range(590,701):
 if i%2==0 or i>=640: print(i,f'rd={(R.from_matrix(T[i,:3,:3])*mr.inv()).magnitude()*180/np.pi:.2f}',f'td={np.linalg.norm(T[i,:3,3]-mt):.3f}')
PY"
```
<details><summary>output</summary>

```
590 rd=8.55 td=0.112
592 rd=7.93 td=0.121
594 rd=7.16 td=0.091
596 rd=6.55 td=0.096
598 rd=5.89 td=0.066
600 rd=5.01 td=0.056
602 rd=4.29 td=0.047
604 rd=4.00 td=0.061
606 rd=3.47 td=0.042
608 rd=3.02 td=0.033
610 rd=2.43 td=0.028
612 rd=2.46 td=0.035
614 rd=1.92 td=0.021
616 rd=1.50 td=0.019
618 rd=1.13 td=0.013
620 rd=0.41 td=0.018
622 rd=0.41 td=0.018
624 rd=0.41 td=0.022
626 rd=0.25 td=0.011
628 rd=0.52 td=0.025
630 rd=0.63 td=0.024
632 rd=0.42 td=0.027
634 rd=0.74 td=0.030
636 rd=0.32 td=0.016
638 rd=0.50 td=0.020
640 rd=0.43 td=0.014
641 rd=0.39 td=0.010
642 rd=0.58 td=0.019
643 rd=0.59 td=0.022
644 rd=0.36 td=0.024
645 rd=0.15 td=0.027
646 rd=0.11 td=0.015
647 rd=0.14 td=0.011
648 rd=0.31 td=0.016
649 rd=0.15 td=0.013
650 rd=0.11 td=0.015
651 rd=0.24 td=0.012
652 rd=0.28 td=0.020
65 …[+1006 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_final_candidate.py <<'PY'
import numpy as np,torch,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';base=np.load(inp+'/reconstruction.npz')['T_world_mesh'].astype(np.float64);F=len(base)
cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);f=np.asarray(g.faces,np.int32)
H,W=576,768;sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy;fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];near,far=.1,100.;ctx=dr.RasterizeCudaContext()
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
o2=np.empty((F,H,W),bool);p2=np.empty((F,H,W),bool)
for i in range(F):o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
def render(TX):
 w=np.c_[v,np.ones(len(v),np.float32)]@TX.astype(np.float32).T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3];x,y,z=c[:,0],c[:,1],c[:,2];clip=np.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,(far+near)/(far-near)*z-2*far*near/(far-near),z],1);rast,_=dr.rasterize(ctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),resolution=[H …[+2649 chars]
```
<details><summary>output</summary>

```
base ALL 0.7690533412371425 8.837978
  initial iou 0.7285366353134171 ch 10.44609 p2o 7.506202 o2p 2.9398873
  manip iou 0.7950267294952108 ch 8.7509985 p2o 6.0536084 o2p 2.6973896
  final iou 0.7849629972554962 ch 6.1753345 p2o 3.8996708 o2p 2.2756639
b0 ALL 0.7711947641578125 8.78219
  initial iou 0.732460174591825 ch 10.323357 p2o 7.5305786 o2p 2.7927787
  manip iou 0.7956663985148553 ch 8.709448 p2o 6.027817 o2p 2.6816316
  final iou 0.7871778616076983 ch 6.207541 p2o 3.89831 o2p 2.3092315
b4 ALL 0.7712092863826101 8.781866
  initial iou 0.732460174591825 ch 10.323357 p2o 7.5305786 o2p 2.7927787
  manip iou 0.795699628782186 ch 8.708706 p2o 6.0275736 o2p 2.6811328
  final iou 0.7871778616076983 ch 6.207541 p2o 3.89831 o2p 2.3092315
b8 ALL 0.7711567525000165 8.780983
  initial iou 0.732 …[+939 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/final_candidates.npz');b=z['base']
for nm in ['b0','b4','b8','b12','b16']:
 s=z[nm]
 def rr(i):return R.from_matrix(s[i-1,:3,:3].T@s[i,:3,:3]).magnitude()*180/np.pi,np.linalg.norm(s[i,:3,3]-s[i-1,:3,3])
 print(nm)
 for i in [279,280,281,618,619,620,621]:print(i,rr(i))
PY"
```
<details><summary>output</summary>

```
b0
279 (0.0, 0.0)
280 (1.9722457478038011, 0.012557166472378185)
281 (0.18235717725309827, 0.005134787952536566)
618 (0.21891877442256408, 0.003306754847165646)
619 (0.22389342470389095, 0.0038598400949441344)
620 (0.887025359191903, 0.0199550342178362)
621 (0.0, 0.0)
b4
279 (0.0, 0.0)
280 (0.28882009621763427, 0.0018389544536356276)
281 (0.6704047204157352, 0.004741174875286255)
618 (0.21891877442256408, 0.003306754847165646)
619 (0.22389342470389095, 0.0038598400949441344)
620 (0.8870253591919025, 0.0199550342178362)
621 (6.387714681259903e-15, 4.47545209131181e-16)
b8
279 (0.0, 0.0)
280 (0.07506084066287341, 0.00047792869110546633)
281 (0.20580790607295363, 0.0014403397523018926)
618 (0.21891877442256408, 0.003306754847165646)
619 (0.22389342470389095, 0.0038598400949441344)
620 (0.8870 …[+655 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');A=z['T_world_mesh'];B=z['T_cam_mesh_foundationpose'];C=z['T_cam_mesh_coconet'];Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for nm,X in [('fp',B),('coco',C)]:
 M=np.einsum('ij,njk->nik',Twc[:3,:3],X[:,:3,:3])
 rel=R.from_matrix(np.einsum('nji,njk->nik',A[:,:3,:3],M)).magnitude()*180/np.pi
 print(nm,rel.mean(),np.median(rel),rel.max(),rel[::50])
 # compare orientation between fp and coco in camera
 rel2=R.from_matrix(np.einsum('nji,njk->nik',B[:,:3,:3],C[:,:3,:3])).magnitude()*180/np.pi
 print('fp-coco',rel2.mean(),rel2.max(),rel2[::50])
PY"
```
<details><summary>output</summary>

```
fp 2.024847565490649 1.7213154653312046 4.276130072544713 [1.00562281 1.50483569 1.44163218 1.22273971 0.49657994 1.20683755
 1.58851478 2.2031146  4.07171859 1.72247269 1.81347431 2.10245898
 2.75263475 3.96044336 3.23260281 3.80388264]
fp-coco 1.3767176574426205 3.660974634743847 [0.18229205 0.48015379 0.76405522 0.5016459  0.77204032 0.18290899
 1.40377637 1.64991519 3.0323722  2.03057517 0.68173527 1.01822435
 1.74540838 2.93093566 2.06691164 2.68085646]
coco 1.2119248267902174 1.2263819499777568 1.242746126105578 [1.18210233 1.18199626 1.18178614 1.18246487 1.18314262 1.18215429
 1.1954257  1.218243   1.23139919 1.23704047 1.24068881 1.2420671
 1.23443059 1.22933447 1.22898664 1.22767032]
fp-coco 1.3767176574426205 3.660974634743847 [0.18229205 0.48015379 0.76405522 0.5016459  0.77204 …[+128 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_rotation_sources.py <<'PY'
import numpy as np,torch,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';z=np.load(inp+'/reconstruction.npz');base=z['T_world_mesh'].astype(np.float64);F=len(base)
cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);f=np.asarray(g.faces,np.int32)
H,W=576,768;sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy;fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];near,far=.1,100.;ctx=dr.RasterizeCudaContext()
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
o2=np.empty((F,H,W),bool);p2=np.empty((F,H,W),bool)
for i in range(F):o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
def render(TX):
 w=np.c_[v,np.ones(len(v),np.float32)]@TX.astype(np.float32).T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3];x,y,z=c[:,0],c[:,1],c[:,2];clip=np.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,(far+near)/(far-near)*z-2*far*near/(far-near),z],1);rast,_=dr.rasterize(ctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),resolution=[H,W]);return (rast[0,:,:,3] …[+2140 chars]
```
<details><summary>output</summary>

```
base ALL 0.7690533412371425 8.837978
  initial 0.7285366353134171 10.44609
  manip 0.7950267294952108 8.7509985
  final 0.7849629972554962 6.1753345
fp ALL 0.769719292461208 8.861088
  initial 0.7198206352110368 10.445647
  manip 0.7972714551565885 8.783421
  final 0.7988578286233502 6.220143
fp0.25 ALL 0.7704763867576637 8.841073
  initial 0.7267240596734802 10.421092
  manip 0.7974809210756459 8.779239
  final 0.7899011330580273 6.1740985
fp0.5 ALL 0.7710360268168407 8.834342
  initial 0.7246705075592678 10.389675
  manip 0.7985476436183725 8.792247
  final 0.7940005564345592 6.1686416
fp0.75 ALL 0.770713258459581 8.829232
  initial 0.7223489217867359 10.409033
  manip 0.7984005785869267 8.7553215
  final 0.7968419003905878 6.1886334
coco ALL 0.765401097086722 8.842117
  initial 0.719289 …[+548 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/rot_overlay.py <<'PY'
import numpy as np,cv2,trimesh,json,h5py
inp='/work/inputs';z=np.load(inp+'/reconstruction.npz');B=z['T_world_mesh'].astype(np.float64);cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices);f=np.asarray(g.faces)
cap=cv2.VideoCapture(inp+'/video.mp4')
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
for fi in [0,100,280,400,600,700,777]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();outs=[]
 for nm,key in [('base',None),('fp','T_cam_mesh_foundationpose'),('coco','T_cam_mesh_coconet')]:
  T=B[fi].copy()
  if key is not None:T[:3,:3]=Twc[:3,:3]@z[key][fi,:3,:3]
  vc=(Tcw@(np.c_[v,np.ones(len(v))]@T.T).T)[:3].T;uv=(K@vc.T).T;uv=uv[:,:2]/uv[:,2:3];m=np.zeros(im.shape[:2],np.uint8);cv2.fillPoly(m,np.round(uv[f]).astype(np.int32),1);ed=cv2.morphologyEx(m,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool);out=im.copy();out[ed]=(0,255,0) if nm=='base' else ((255,0,255) if nm=='fp' else (0,255,255));cv2.rectangle(out,(0,0),(130,45),(0,0,0),-1);cv2.putText(out,nm,(5,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);outs.append(out)
 # combined
 out=im.copy()
 for nm,key,col in [('base',None,(0,255,0)),('fp','T_cam_mesh_foundationpose',(255,0,255)),('coco','T_cam_mesh_coconet',(0,255,255))]:
  T=B[fi].copy()
  if key is not …[+458 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
for fi in [0,100,280,400,600,700,777]:
 im=cv2.imread(f'/work/scratch/rot_ov{fi}.jpg')
 # 4 panels width 1536 each, height1152. Crop object region based fi approximate center
 if fi<300: y0,y1,x0,x1=650,1000,450,950
 else: y0,y1,x0,x1=650,1000,650,1150
 crops=[im[y0:y1,i*1536+x0:i*1536+x1] for i in range(4)]
 cv2.imwrite(f'/work/scratch/rot_ov{fi}_crop.jpg',np.concatenate(crops,1))
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
im=cv2.imread('/work/scratch/rot_ov700.jpg')
# base [0], fp[1], coco[2], combined[3], full 6144x1152
for idx,nm in [(0,'base'),(1,'fp'),(2,'coco'),(3,'comb')]:
 c=im[650:1000,idx*1536+650:idx*1536+1150]
 c=cv2.resize(c,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
 cv2.imwrite(f'/work/scratch/final_{nm}_3x.png',c)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_final_candidate.py';s=open(p).read()
s=s.replace(\"for i in range(A,min(A+bw,B)):\\n  t=0.5-0.5*np.cos(np.pi*(i-A+1)/bw)\\n  out[i,:3,:3]=slerp(out[A-1,:3,:3],out[i,:3,:3],t);out[i,:3,3]=(1-t)*out[A-1,:3,3]+t*out[i,:3,3]\\n for i in range(B,min(B+bw,F)):\\n  t=0.5-0.5*np.cos(np.pi*(i-B)/bw)\\n  out[i,:3,:3]=slerp(out[i,:3,:3],out[B,:3,:3],t);out[i,:3,3]=(1-t)*out[i,:3,3]+t*out[B,:3,3]\", \"for i in range(A,min(A+bw,B)):\\n  t=0.5-0.5*np.cos(np.pi*(i-A+1)/bw)\\n  out[i,:3,:3]=slerp(out[A-1,:3,:3],out[i,:3,:3],t);out[i,:3,3]=(1-t)*out[A-1,:3,3]+t*out[i,:3,3]\\n for i in range(B,min(B+bw,F)):\\n  t=0.5-0.5*np.cos(np.pi*(i-B+1)/bw)\\n  out[i,:3,:3]=slerp(out[i,:3,:3],out[B,:3,:3],t);out[i,:3,3]=(1-t)*out[i,:3,3]+t*out[B,:3,3]\")
s=s.replace(\"for bw in [0,4,8,12,16,20]\",\"for bw in [4,8,12,16]\")
# avoid overwriting final_candidates? okay
open('/work/scratch/eval_final_candidate2.py','w').write(s)
PY
python /work/scratch/eval_final_candidate2.py"
```
<details><summary>output</summary>

```
base ALL 0.7690533412371425 8.837978
  initial iou 0.7285366353134171 ch 10.44609 p2o 7.506202 o2p 2.9398873
  manip iou 0.7950267294952108 ch 8.7509985 p2o 6.0536084 o2p 2.6973896
  final iou 0.7849629972554962 ch 6.1753345 p2o 3.8996708 o2p 2.2756639
b4 ALL 0.7712092863826101 8.781866
  initial iou 0.732460174591825 ch 10.323357 p2o 7.5305786 o2p 2.7927787
  manip iou 0.795699628782186 ch 8.708706 p2o 6.0275736 o2p 2.6811328
  final iou 0.7871778616076983 ch 6.207541 p2o 3.89831 o2p 2.3092315
b8 ALL 0.7711567525000165 8.780983
  initial iou 0.732460174591825 ch 10.323357 p2o 7.5305786 o2p 2.7927787
  manip iou 0.7955794188978984 ch 8.706687 p2o 6.0265436 o2p 2.6801429
  final iou 0.7871778616076983 ch 6.207541 p2o 3.89831 o2p 2.3092315
b12 ALL 0.7710340423618797 8.78202
  initial iou 0.7 …[+445 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_radial_compare.py <<'PY'
import numpy as np,cv2,trimesh,json,h5py
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices);f=np.asarray(g.faces)
rr=R.from_matrix(T[620:,:3,:3]);cm=np.eye(4);cm[:3,:3]=rr.mean().as_matrix();cm[:3,3]=T[620:,:3,3].mean(0);ray=cm[:3,3]-Twc[:3,3];ray/=np.linalg.norm(ray)
cap=cv2.VideoCapture(inp+'/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,700);ok,im=cap.read();cap.release()
cols=[(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255)]
for al in [0.05,0,-0.05,-0.10,-0.15]:
 cc=cm.copy();cc[:3,3]+=al*ray;vc=(Tcw@(np.c_[v,np.ones(len(v))]@cc.T).T)[:3].T;uv=(K@vc.T).T;uv=uv[:,:2]/uv[:,2:3];m=np.zeros(im.shape[:2],np.uint8);cv2.fillPoly(m,np.round(uv[f]).astype(np.int32),1);ed=cv2.morphologyEx(m,cv2.MORPH_GRADIENT,np.ones((2,2),np.uint8)).astype(bool);out=im.copy();out[ed]=cols[0];cv2.putText(out,f'{al:+.2f}',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),3);cv2.imwrite(f'/work/scratch/rad_{abs(int(al*100))}.jpg',out[700:1010,760:1080])
PY
python /work/scratch/final_radial_compare.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/rad_{abs(int(al\\*100))}/rad_{'p' if al>0 else 'm' if al<0 else 'z'}{abs(int(al*100))}/\" /work/scratch/final_radial_compare.py
python /work/scratch/final_radial_compare.py
python - <<'PY'
import cv2,numpy as np
ims=[cv2.imread('/work/scratch/rad_p5.jpg'),cv2.imread('/work/scratch/rad_z0.jpg'),cv2.imread('/work/scratch/rad_m5.jpg'),cv2.imread('/work/scratch/rad_m10.jpg'),cv2.imread('/work/scratch/rad_m15.jpg')]
cv2.imwrite('/work/scratch/rad_compare_all.jpg',np.concatenate(ims,0))
PY"
```
<details><summary>output</summary>

```
[ WARN:0@0.031] global loadsave.cpp:268 findDecoder imread_('/work/scratch/rad_p5.jpg'): can't open/read file: check file path/integrity
[ WARN:0@0.031] global loadsave.cpp:268 findDecoder imread_('/work/scratch/rad_z0.jpg'): can't open/read file: check file path/integrity
[ WARN:0@0.031] global loadsave.cpp:268 findDecoder imread_('/work/scratch/rad_m5.jpg'): can't open/read file: check file path/integrity
[ WARN:0@0.031] global loadsave.cpp:268 findDecoder imread_('/work/scratch/rad_m10.jpg'): can't open/read file: check file path/integrity
[ WARN:0@0.032] global loadsave.cpp:268 findDecoder imread_('/work/scratch/rad_m15.jpg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
ValueError: zero-dimensional arrays cann …[+19 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/final_radial_compare.py';s=open(p).read();s=s.replace(\"cv2.imwrite(f'/work/scratch/rad_{'p' if al>0 else 'm' if al<0 else 'z'}{abs(int(al*100))}.jpg'\",\"tag = ('p' if al>0 else ('m' if al<0 else 'z')) + str(abs(int(al*100)))\\n cv2.imwrite(f'/work/scratch/rad_{tag}.jpg'\");open(p,'w').write(s)
PY
python /work/scratch/final_radial_compare.py
python - <<'PY'
import cv2,numpy as np
ims=[cv2.imread('/work/scratch/rad_p5.jpg'),cv2.imread('/work/scratch/rad_z0.jpg'),cv2.imread('/work/scratch/rad_m5.jpg'),cv2.imread('/work/scratch/rad_m10.jpg'),cv2.imread('/work/scratch/rad_m15.jpg')]
cv2.imwrite('/work/scratch/rad_compare_all.jpg',np.concatenate(ims,0))
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
paths=['rad_p5.jpg','rad_z0.jpg','rad_m5.jpg','rad_m10.jpg','rad_m15.jpg']
ims=[cv2.resize(cv2.imread('/work/scratch/'+p),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST) for p in paths]
cv2.imwrite('/work/scratch/rad_compare_h.jpg',np.concatenate(ims,1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/depth_vis.py <<'PY'
import numpy as np,h5py,cv2,torch,trimesh,json
import nvdiffrast.torch as dr
inp='/work/inputs';T=np.load(inp+'/reconstruction.npz')['T_world_mesh'];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);f=np.asarray(g.faces,np.int32)
H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];near,far=.1,100.;ctx=dr.RasterizeCudaContext()
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as h:al=h['aligned/front_stereo_camera_left'][:]
cap=cv2.VideoCapture(inp+'/video.mp4')
for i in [0,280,400,600,700,777]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();dm=cv2.imdecode(al[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
 vh=np.c_[v,np.ones(len(v),np.float32)];w=vh@T[i].astype(np.float32).T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3];x,y,z=c[:,0],c[:,1],c[:,2];clip=np.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,(far+near)/(far-near)*z-2*far*near/(far-near),z],1);rast,_=dr.rasterize(ctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),resolution=[H,W]);zw,_=dr.interpolate(torch.from_numpy((clip[:,2:3]/clip[:,3:4]))[None].cuda(),rast,torch.from_numpy(f).cuda());ndc=zw[0,:,:,0].cpu().numpy();zc=2*far*near/((far+near)-(far-near)*ndc);mask=(rast[0,:,:,3]>0).cpu().numpy()
 # cr …[+742 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.signal import savgol_filter
r=np.load('/work/scratch/depth_residuals.npy')[:,2]
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
ray=T[:,:3,3]-Twc[:3,3];ray/=np.linalg.norm(ray,axis=1)[:,None];raycam=ray@Tcw[:3,:3].T;alpha=np.clip(r/raycam[:,2],-.3,.3)
print('alpha quant',np.percentile(alpha,[0,5,25,50,75,95,100]))
for a,b in [(0,280),(280,620),(620,778)]:
 print(a,b,'r med mean',np.median(r[a:b]),r[a:b].mean(),'alpha med mean std',np.median(alpha[a:b]),alpha[a:b].mean(),alpha[a:b].std())
 for w in [15,31,61]:
  s=savgol_filter(alpha,w,2,mode='interp');print(' w',w,'median phase',np.median(s[a:b]),'mean',s[a:b].mean())
for i in range(0,778,20):print(i,r[i],alpha[i])
PY"
```
<details><summary>output</summary>

```
alpha quant [-0.26188724 -0.21011935 -0.10377753 -0.02394719  0.05758566  0.15110576
  0.20790084]
0 280 r med mean 0.035485267639160156 0.04371976980141231 alpha med mean std 0.03745347197894407 0.0461568687800261 0.06682649452127017
 w 15 median phase 0.03516401976058486 mean 0.046166136246912605
 w 31 median phase 0.03663615646628562 mean 0.04621686621931539
 w 61 median phase 0.03886450000549062 mean 0.046374390995478194
280 620 r med mean -0.04155954718589783 -0.0462696257759543 alpha med mean std -0.04236300171099323 -0.048243694011970366 0.11993069014434472
 w 15 median phase -0.041754218058367425 mean -0.04823306800635466
 w 31 median phase -0.04205232307587886 mean -0.04833146364294303
 w 61 median phase -0.04029772851358211 mean -0.04884819592320066
620 778 r med mean -0.07807844 …[+2068 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_depth_candidates.py <<'PY'
import numpy as np,torch,trimesh,json,h5py,cv2
import nvdiffrast.torch as dr
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';base=np.load(inp+'/reconstruction.npz')['T_world_mesh'].astype(np.float64);F=len(base)
cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);f=np.asarray(g.faces,np.int32)
H,W=576,768;sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy;fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];near,far=.1,100.;ctx=dr.RasterizeCudaContext()
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
o2=np.empty((F,H,W),bool);p2=np.empty((F,H,W),bool)
for i in range(F):o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
def render(TX):
 w=np.c_[v,np.ones(len(v),np.float32)]@TX.astype(np.float32).T;c=(np.c_[w[:,:3],np.ones(len(v),np.float32)]@Tcw.T)[:,:3];x,y,z=c[:,0],c[:,1],c[:,2];clip=np.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,(far+near)/(far-near)*z-2*far*near/(far-near),z],1);rast,_=dr.rasterize(ctx,torch.from_numpy(clip)[None].cuda(),torch.from_numpy(f).cuda(),reso …[+3413 chars]
```
<details><summary>output</summary>

```

 smooth ALL iou 0.7711567525000165 ch 8.780983 edge 1.2763281
init iou 0.732460174591825 ch 10.323357 edge 1.6854964
move iou 0.7955794188978984 ch 8.706687 edge 1.0471569
final iou 0.7871778616076983 ch 6.207541 edge 1.0472723

 d25 ALL iou 0.7769014660320892 ch 8.874729 edge 1.2649307
init iou 0.7225550179520819 ch 11.058526 edge 1.697848
move iou 0.8043843073004847 ch 8.57152 edge 1.0410838
final iou 0.8140713358494797 ch 5.6571856 edge 0.9830004

 d50 ALL iou 0.7838598731763778 ch 8.520945 edge 1.2420212
init iou 0.7181552251961661 ch 10.448097 edge 1.6485231
move iou 0.8119859455443352 ch 8.429001 edge 1.0419062
final iou 0.8397740303241865 ch 5.30359 edge 0.95588684

 edge ALL iou 0.7875048966405287 ch 8.642833 edge 1.2467022
init iou 0.7212009686819962 ch 10.87768 edge 1.6951256
mo …[+111 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"residual\\|alpha\\|ts61\\|cands\" -A20 -B2 /work/scratch/eval_depth_candidates2.py | tail -80"
```
<details><summary>output</summary>

```
36-def radial(TX,offs):
37- out=TX.copy();rays=out[:,:3,3]-Twc[:3,3];rays/=np.linalg.norm(rays,axis=1)[:,None];out[:,:3,3]+=rays*np.asarray(offs)[:,None];return out
38:cands={'smooth':smooth}
39-for nm,off in [('d25',[.0187,-.021,-.04]),('d50',[.0375,-.042,-.10]),('edge',[.025,-.04,-.12])]:
40- oo=np.zeros(F)
41- for (a,b),q in zip([(0,280),(280,620),(620,F)],off):oo[a:b]=q
42: cands[nm]=radial(smooth,oo)
43:# residual-derived low-frequency correction, with separate phase anchors
44:r=np.load('/work/scratch/depth_residuals.npy')[:,2]
45-rays=base[:,:3,3]-Twc[:3,3];rays/=np.linalg.norm(rays,axis=1)[:,None];rc=rays@Tcw[:3,:3].T
46:alpha=np.clip(r/rc[:,2],-.3,.3)
47-for wl,damp in [(61,.5),(101,.5),(61,1.0)]:
48: aa=savgol_filter(alpha,wl,2,mode='interp');oo=np.zeros(F)
49- for (a,b),anchor i …[+1356 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_depth_candidates.py';s=open(p).read()
old=\"\"\"cands={'smooth':smooth}
for nm,off in [('d25',[.0187,-.021,-.04]),('d50',[.0375,-.042,-.0796]),('edge',[.025,-.04,-.12])]:
 oo=np.zeros(F)
 for (a,b),q in zip([(0,280),(280,620),(620,F)],off):oo[a:b]=q
 cands[nm]=radial(smooth,oo)\"\"\"
new=\"\"\"cands={'smooth':smooth}
for nm,off in [('d25',[.0187,-.021,-.04]),('d50',[.0375,-.042,-.10]),('edge',[.025,-.04,-.12])]:
 oo=np.zeros(F)
 for (a,b),q in zip([(0,280),(280,620),(620,F)],off):oo[a:b]=q
 cands[nm]=radial(smooth,oo)
# residual-derived low-frequency correction, with separate phase anchors
r=np.load('/work/scratch/depth_residuals.npy')[:,2]
rays=base[:,:3,3]-Twc[:3,3];rays/=np.linalg.norm(rays,axis=1)[:,None];rc=rays@Tcw[:3,:3].T
alpha=np.clip(r/rc[:,2],-.3,.3)
for wl,damp in [(61,.5),(101,.5),(61,1.0)]:
 aa=savgol_filter(alpha,wl,2,mode='interp');oo=np.zeros(F)
 for (a,b),anchor in zip([(0,280),(280,620),(620,F)],[.0375,-.042,-.10]):
  aa[a:b]=anchor+damp*(aa[a:b]-np.median(aa[a:b]))
 cands[f'ts{wl}_{damp}']=radial(smooth,oo)\"\"\"
s=s.replace(old,new);open('/work/scratch/eval_depth_candidates2.py','w').write(s)
PY
python /work/scratch/eval_depth_candidates2.py"
```
<details><summary>output</summary>

```

 smooth ALL iou 0.7711567525000165 ch 8.780983 edge 1.2763281
init iou 0.732460174591825 ch 10.323357 edge 1.6854964
move iou 0.7955794188978984 ch 8.706687 edge 1.0471569
final iou 0.7871778616076983 ch 6.207541 edge 1.0472723

 d25 ALL iou 0.7769014660320892 ch 8.874729 edge 1.2649307
init iou 0.7225550179520819 ch 11.058526 edge 1.697848
move iou 0.8043843073004847 ch 8.57152 edge 1.0410838
final iou 0.8140713358494797 ch 5.6571856 edge 0.9830004

 d50 ALL iou 0.7858176415376256 ch 8.4954195 edge 1.2372831
init iou 0.7181552251961661 ch 10.448097 edge 1.6485231
move iou 0.8119859455443352 ch 8.429001 edge 1.0419062
final iou 0.8494141808624818 ch 5.1779013 edge 0.9327887

 edge ALL iou 0.7875048966405287 ch 8.642833 edge 1.2467022
init iou 0.7212009686819962 ch 10.87768 edge 1.6951256
 …[+807 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_depth_candidates2.py').read()
s=s.replace(\"for nm,off in [('d25',[.0187,-.021,-.04]),('d50',[.0375,-.042,-.10]),('edge',[.025,-.04,-.12])]:\\n oo=np.zeros(F)\\n for (a,b),q in zip([(0,280),(280,620),(620,F)],off):oo[a:b]=q\\n cands[nm]=radial(smooth,oo)\\n\",\"\")
s=s.replace(\"aa=savgol_filter(alpha,wl,2,mode='interp');oo=np.zeros(F)\\n for (a,b),anchor in zip([(0,280),(280,620),(620,F)],[.0375,-.042,-.10]):\\n  aa[a:b]=anchor+damp*(aa[a:b]-np.median(aa[a:b]))\\n cands[f'ts{wl}_{damp}']=radial(smooth,oo)\",\"aa=savgol_filter(alpha,wl,2,mode='interp');oo=np.zeros(F)\\n for (a,b),anchor in zip([(0,280),(280,620),(620,F)],[.0375,-.042,-.10]):\\n  oo[a:b]=anchor+damp*(aa[a:b]-np.median(aa[a:b]))\\n cands[f'ts{wl}_{damp}']=radial(smooth,oo)\")
open('/work/scratch/eval_depth_ts.py','w').write(s)
PY
python /work/scratch/eval_depth_ts.py"
```
<details><summary>output</summary>

```

 smooth ALL iou 0.7711567525000165 ch 8.780983 edge 1.2763281
init iou 0.732460174591825 ch 10.323357 edge 1.6854964
move iou 0.7955794188978984 ch 8.706687 edge 1.0471569
final iou 0.7871778616076983 ch 6.207541 edge 1.0472723

 ts61_0.5 ALL iou 0.7836945907110515 ch 8.583508 edge 1.2416818
init iou 0.713139045206287 ch 10.845629 edge 1.6809131
move iou 0.8111222050916153 ch 8.33992 edge 1.0393124
final iou 0.8497082859765098 ch 5.098862 edge 0.90306187

 ts101_0.5 ALL iou 0.7845446746165252 ch 8.56483 edge 1.2418622
init iou 0.7144965861027955 ch 10.800274 edge 1.6789678
move iou 0.8121554979756892 ch 8.319689 edge 1.0411168
final iou 0.8492648318426551 ch 5.1307993 edge 0.9035108

 ts61_1.0 ALL iou 0.773847711155408 ch 8.705211 edge 1.2896723
init iou 0.7082331277795036 ch 10.846242 ed …[+124 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_ramped.py <<'PY'
exec(open('/work/scratch/eval_depth_candidates2.py').read().split(\"cands={'smooth':smooth}\")[0])
# smooth phase offsets
phase=np.zeros(F)
for a,b,q in [(0,280,.0375),(280,620,-.042),(620,F,-.12)]:phase[a:b]=q
ramp=phase.copy();tw=20
for i in range(280,280+tw):
 t=.5-.5*np.cos(np.pi*(i-279)/tw);ramp[i]=(1-t)*.0375+t*(-.042)
for i in range(620,620+tw):
 t=.5-.5*np.cos(np.pi*(i-619)/tw);ramp[i]=(1-t)*(-.042)+t*(-.12)
cands={'ramp':radial(smooth,ramp),'step':radial(smooth,phase)}
# sequential images
cap=cv2.VideoCapture(inp+'/video.mp4');ims={}
for i in range(F):
 ok,im=cap.read();assert ok
 if i%5==0:ims[i]=cv2.resize(im,(W,H))
cap.release()
for name,TX in cands.items():
 ious=[];pds=[];ods=[];edgep=[]
 for i in range(F):
  pred=render(TX[i]);q=~p2[i];inter=(pred&o2[i]&q).sum();uni=((pred|o2[i])&q).sum();ious.append(inter/uni);pd,od=cham(pred,o2[i],p2[i]);pds.append(pd);ods.append(od)
  if i in ims:
   gray=cv2.cvtColor(ims[i],cv2.COLOR_BGR2GRAY);ed=cv2.Canny(gray,60,140);dt=cv2.distanceTransform((ed==0).astype(np.uint8),cv2.DIST_L2,5);pc=contour(pred);bad=cv2.dilate(p2[i].astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool);d=np.array([dt[y,x] for x,y in pc if not bad[y,x]]);edgep.append(d.mean() if len(d) else np.nan)
 ious=np.array(ious);pds=np.array(pds);ods=np.array(ods);edgep=np.array(edgep);ek=np.array(sorted(ims))
 print('\\n',name,'ALL iou',ious.mean(),'ch',np.nanmean(pds+ods),'edge',np.nanmean(edgep))
 f …[+220 chars]
```
<details><summary>output</summary>

```

 ramp ALL iou 0.785925367827741 ch 8.483832 edge 1.2306746
init iou 0.7181552251961661 ch 10.448097 edge 1.6485231
move iou 0.8111170025390674 ch 8.445271 edge 1.0411019
final iou 0.8518145079226142 ch 5.08584 edge 0.90228224

 step ALL iou 0.7866950981563553 ch 8.46792 edge 1.2292435
init iou 0.7181552251961661 ch 10.448097 edge 1.6485231
move iou 0.8119859455443352 ch 8.429001 edge 1.0419062
final iou 0.8537348216812909 ch 5.0424957 edge 0.89359677

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/outputs && cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF world trajectory for the stool.

The released baseline is already visually well registered frame by frame, but it
contains tracking jitter and a depth bias that grows during the manipulation.
This solver keeps the baseline where it is trustworthy and applies three
trajectory-level operations:

1. Pool every frame in the two observed rest phases into one exact SE(3) pose.
2. Smooth the manipulated phase in a local SO(3) tangent space with a short
   Savitzky-Golay filter, retaining the baseline'"'s motion.
3. Correct the depth along each camera ray using phase-pooled MoGe depth
   residuals and silhouette-edge validation.

The phase boundaries and depth offsets below are explicit validation constants.
They were selected from the video/masks, not from ground truth.
\"\"\"

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# Video inspection: the object is at rest through frame 279, is lifted/tilted
# from 280 to 619, and remains upright from frame 620 onward.
REST0_END = 280
REST1_START = 620

# 0.57 s local quadratic fit. This removes baseline per-frame tracking jitter
# while preserving the low-frequency transport and tipping motion.
SMOOTH_WINDOW = 17
SMOOTH_POLY = 2

# Eight-frame smoothstep transiti …[+7334 chars]
```
<details><summary>output</summary>

```
['T_world_mesh']
(778, 4, 4) float64 True -0.776900020208516 3.1653737239112947
change rot mean/med/max 0.4275794857423381 0.3770878357761961 1.8846515609347343
change trans mean/med/max 0.05680527192732144 0.04552469448680799 0.15961332979699394
rest0 rot 0.5800685466662796 trans 0.0427969461024915
move rot 0.310625558624584 trans 0.04110763253423902
rest1 rot 0.4090187146115327 trans 0.11540988347542533

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
z=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];b=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.c_[g.vertices,np.ones(len(g.vertices))];pl=json.load(open('/work/inputs/ground_plane.json'))['plane'];n=np.array(pl[:3]);d=pl[3]
def gap(T):return (np.einsum('fvi,i->fv',np.einsum('fij,vj->fvi',T,v)[:,:,:3],n)+d).min(1)
for nm,T in [('base',b),('out',z)]:
 q=gap(T);print(nm)
 for a,bb,s in [(0,280,'rest0'),(280,620,'move'),(620,778,'rest1')]:print(s,np.median(q[a:bb]),q[a:bb].mean(),q[a:bb].min(),q[a:bb].max())
PY"
```
<details><summary>output</summary>

```
base
rest0 -0.028878949502803675 -0.03182890334876453 -0.060342862792663876 -0.010401574342787434
move -0.02059162009414839 -0.019121050224571178 -0.0421152087385428 0.0032946283344421623
rest1 -0.025191104946753473 -0.025178961699645997 -0.04081681061937714 -0.009341991359651969
out
rest0 -0.046520252098078974 -0.046520252098078974 -0.046520252098078974 -0.046520252098078974
move -0.00842522502184917 -0.007422227772096604 -0.045755435156283664 0.010239509830617166
rest1 0.006693980815468015 0.005447969664630083 -0.01390158393242591 0.006693980815468015

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];b=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
print('base mean t',b[620:,:3,3].mean(0),'out const t',T[700,:3,3])
print('base final min y',b[700,1,3])
pl=json.load(open('/work/inputs/ground_plane.json'))['plane'];print('floor y approx',-pl[3]/pl[1])
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.c_[g.vertices,np.ones(len(g.vertices))];n=np.array(pl[:3]);d=pl[3]
for nm,X in [('b',b[700]),('o',T[700])]:
 w=(X@v.T)[:3].T;sd=w@n+d;print(nm,'t',X[:3,3],'min',sd.min(),'arg',sd.argmin(),'y',w[sd.argmin(),1])
PY"
```
<details><summary>output</summary>

```
base mean t [0.45591152 0.6898051  3.206583  ] out const t [0.44127561 0.6646132  3.09017335]
base final min y 0.6833642
floor y approx 1.3880199779972688
b t [0.4550219 0.6833642 3.1987884] min -0.019756171204128536 arg 5977 y 1.2073566507672817
o t [0.44127561 0.6646132  3.09017335] min 0.006693980815468015 arg 5977 y 1.187010015810201

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.c_[g.vertices,np.ones(len(g.vertices))];pl=json.load(open('/work/inputs/ground_plane.json'))['plane'];n=np.array(pl[:3]);d=pl[3]
for a,b in [(620,778),(650,778)]:
 m=np.eye(4);m[:3,:3]=R.from_matrix(T[a:b,:3,:3]).mean().as_matrix();m[:3,3]=T[a:b,:3,3].mean(0);w=(m@v.T)[:3].T;print(a,b,m[:3,3],(w@n+d).min())
PY"
```
<details><summary>output</summary>

```
620 778 [0.45591152 0.68980509 3.20658302] -0.025187895543201888
650 778 [0.45569938 0.69052094 3.20411801] -0.02547360627882833

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/validate_final.py <<'PY'
import json
import numpy as np, torch, trimesh, cv2, h5py
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
inp='/work/inputs';base=np.load(inp+'/reconstruction.npz')['T_world_mesh'].astype(np.float64);result=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(np.float64);methods={'baseline':base,'result':result};F=len(base)
cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32);g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices,np.float32);fv=np.asarray(g.faces,np.int32)
H,W=576,768;sx=W/1536;sy=H/1152;Ks=K.copy();Ks[0]*=sx;Ks[1]*=sy;fx,fy=Ks[0,0],Ks[1,1];cx,cy=Ks[0,2],Ks[1,2];near,far=.1,100.;ctx=dr.RasterizeCudaContext()
with h5py.File(inp+'/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as h:aligned=h['aligned/front_stereo_camera_left'][:]
o2=np.empty((F,H,W),bool);p2=np.empty((F,H,W),bool)
for i in range(F):o2[i]=cv2.resize(om[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p2[i]=cv2.resize(pm[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
cap=cv2.VideoCapture(inp+'/video.mp4');ims={}
for i in range(F):
 ok,im=cap.read();assert ok
 if i%5==0:ims[i]=cv2.resize(im,(W,H))
cap.release()
def render(TX …[+3767 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/validate_final.py", line 46, in <module>
    rr=R.from_matrix(T[:,:3,:3]);rstep=(R.from_matrix(np.einsum('nji,njk->nik',rr[:-1],rr[1:])).magnitude()*180/np.pi);acc=np.linalg.norm(T[2:,:3,3]-2*T[1:-1,:3,3]+T[:-2,:3,3],axis=1)
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/einsumfunc.py", line 1371, in einsum
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: einstein sum subscripts string contains too many subscripts for operand 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/rr=R.from_matrix(T\\[:,:3,:3\\]);rstep=(R.from_matrix(np.einsum('nji,njk->nik',rr\\[:-1\\],rr\\[1:\\])).magnitude()\\*180\\/np.pi)/rr=T[:,:3,:3];rstep=(R.from_matrix(np.einsum('nji,njk->nik',rr[:-1],rr[1:])).magnitude()*180\\/np.pi)/\" /work/scratch/validate_final.py
python /work/scratch/validate_final.py"
```
<details><summary>output</summary>

```
{
  "baseline": {
    "all": {
      "mask_iou": 0.7690533412371425,
      "boundary_chamfer_px": 8.83797836303711,
      "image_edge_distance_px": 1.3016173839569092,
      "median_depth_residual_m": 0.00827854871749878,
      "floor_gap_median_m": -0.024915431262435894,
      "rotation_step_deg_mean": 0.3968182659453651,
      "translation_accel_m_mean": 0.01504989070494929
    },
    "phases": {
      "rest0": {
        "mask_iou": 0.7285366353134171,
        "boundary_chamfer_px": 10.446089744567871,
        "image_edge_distance_px": 1.7334129810333252,
        "median_depth_residual_m": 0.08902883529663086,
        "depth_residual_mad_m": 0.08902883529663086,
        "floor_gap_median_m": -0.028878949502803675,
        "rotation_step_deg_mean": 0.367145875134033,
        "translation_ …[+2671 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_overlay.py <<'PY'
import numpy as np,cv2,trimesh,json,h5py
inp='/work/inputs';b=np.load(inp+'/reconstruction.npz')['T_world_mesh'];r=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];cam=json.load(open(inp+'/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load(inp+'/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(g.vertices);f=np.asarray(g.faces)
S=.5;W=768;H=576;Ks=K.copy();Ks[:2]*=S;cap=cv2.VideoCapture(inp+'/video.mp4');frames=[]
for i in [0,150,279,300,400,500,619,650,700,777]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.resize(im,(W,H));out=im
 for T,col in [(b[i],(0,255,0)),(r[i],(255,0,255))]:
  vc=(Tcw@(np.c_[v,np.ones(len(v))]@T.T).T)[:3].T;uv=(K@vc.T).T;uv=uv[:,:2]/uv[:,2:3]*S;m=np.zeros((H,W),np.uint8);cv2.fillPoly(m,np.round(uv[f]).astype(np.int32),1);ed=cv2.morphologyEx(m,cv2.MORPH_GRADIENT,np.ones((2,2),np.uint8)).astype(bool);out[ed]=col
 cv2.rectangle(out,(0,0),(100,35),(0,0,0),-1);cv2.putText(out,str(i),(5,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);frames.append(out)
cap.release()
while len(frames)%5:frames.append(np.zeros_like(frames[0]))
rows=[np.concatenate(frames[j:j+5],1) for j in range(0,len(frames),5)];cv2.imwrite('/work/scratch/final_overlay.jpg',np.concatenate(rows,0))
PY
python /work/scratch/final_overlay.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "/usr/bin/time -f 'elapsed=%e maxrss_kb=%M' python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "TIMEFORMAT='elapsed=%3R user=%3U sys=%3S'; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
elapsed=0.639 user=4.670 sys=3.279

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy,scipy,cv2,h5py,trimesh,torch,nvdiffrast
mods=[('python',sys.version.split()[0]),('numpy',numpy.__version__),('scipy',scipy.__version__),('opencv-python',cv2.__version__),('h5py',h5py.__version__),('trimesh',trimesh.__version__),('torch',torch.__version__),('nvdiffrast',getattr(nvdiffrast,'__version__','installed (version attribute unavailable)'))]
for a,b in mods:print(a,b)
PY
ffmpeg -version | head -1
ffprobe -version | head -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```
<details><summary>output</summary>

```
python 3.11.10
numpy 1.26.3
scipy 1.16.3
opencv-python 4.11.0
h5py 3.16.0
trimesh 5.1.1
torch 2.5.1+cu124
nvdiffrast 0.4.0
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
ffprobe version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2007-2021 the FFmpeg developers
NVIDIA RTX 6000 Ada Generation, 49140 MiB
NVIDIA RTX 6000 Ada Generation, 49140 MiB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Tall bar stool trajectory recovery

## Result

"'`object_poses_world.npz` contains `T_world_mesh` as a finite `(778, 4, 4)`
`float64` array, mapping the supplied mesh-file coordinates into the released
world frame.

The baseline was already a strong per-frame registration: its rendered
silhouette and image edges follow the stool closely, and its rotation sequence
contains no sudden 180 degree segment flip. I therefore made only changes that
had trajectory-level support rather than replacing the baseline wholesale.

## Phases found

| Frames | Phase | Evidence / treatment |
|---|---|---|
| 0-279 | stool lying at rest | The object mask is stationary. All 280 baseline poses are pooled into one rigid SE(3) pose. |
| 280-619 | two-handed lift and tip-up | The stool motion begins near frame 280. Rotation is smoothed in a local SO(3) tangent space and translation with a 17-frame quadratic Savitzky-Golay fit. |
| 620-777 | stool upright at rest | Rotation reaches its settled value by about frame 620; the human releases the stool at about frame 652. All frames 620-777 are pooled into one rigid pose. |

Eight-frame smoothstep pose transitions join the pooled rest poses to the
smoothed manipulation trajectory. A 20-frame smoothstep ramps the depth
correction across each phase boundary.

## What changed

1. **Static pose pooling.** Per-frame tracker noise in each rest phase was
   replaced by the chordal mean rotation and mean tran …[+6786 chars]
```
<details><summary>output</summary>

```
output check OK (778, 4, 4) float64

```
</details>

**Agent:** Completed the task and produced all required outputs under `/work/outputs`:

- `object_poses_world.npz` — `T_world_mesh`, `(778, 4, 4)`, `float64`, finite
- `solve.py` — deterministic offline solver, runtime ~0.64 s
- `REPORT.md` — phases, method, validation table, confidence, limitations
- `TOOLS.md` — tools, versions, and dependency disclosure

Key validation improvements over the baseline:

- SAM mask IoU: `0.7691 → 0.7859`
- Outer-boundary chamfer: `8.838 → 8.484 px`
- Image-edge distance: `1.302 → 1.231 px`
- Mean rotation step: `0.397 → 0.129°/frame`
- Median floor gap magnitude: `2.49 → 1.29 cm`

The final trajectory pools two static phases, smooths the manipulation phase, applies validated camera-ray depth correction, and preserves rotation continuity without introducing a symmetry flip.
