"""Optional audit and visualizations; not needed by solve.py."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('/work/inputs')
out = Path('/work/outputs')
r = np.load(root / 'reconstruction.npz')
b = r['T_world_mesh'].astype(np.float64)
p = np.load(out / 'object_poses_world.npz')['T_world_mesh']
assert p.shape == (553, 4, 4) and p.dtype == np.float64
assert np.isfinite(p).all() and np.array_equal(p[:, :3, :3], b[:, :3, :3])
assert np.array_equal(p[:, 3], b[:, 3])
c = json.loads((root / 'camera.json').read_text()); W = np.array(c['T_world_cam']); K = np.array(c['K'])
bc = np.linalg.inv(W)[None] @ b; pc = np.linalg.inv(W)[None] @ p
mesh = trimesh.load(root / 'object_mesh.glb', force='scene', process=False).to_geometry()
hull = ConvexHull(mesh.vertices)
planes = hull.equations

def front_depth(T, x, y):
    rays = np.column_stack([(x-K[0,2])/K[0,0], (y-K[1,2])/K[1,1], np.ones(len(x))]) @ T[:3,:3]
    origin = -T[:3,3] @ T[:3,:3]
    nd = rays @ planes[:,:3].T
    no = origin @ planes[:,:3].T + planes[:,3]
    with np.errstate(divide='ignore', invalid='ignore'):
        depths = -no[None] / nd
    entry = np.max(np.where(nd < -1.e-10, depths, -np.inf), axis=1)
    exit = np.min(np.where(nd > 1.e-10, depths, np.inf), axis=1)
    return np.where((entry <= exit) & (entry > 0), entry, np.nan)

evidence = {'mesh_extents_m': mesh.extents.tolist(), 'sampled_depth': []}
with h5py.File(root/'masks.h5') as masks, h5py.File('/pipeline/depth/moge2/aligned.h5') as depth:
    for f in np.unique(np.r_[np.arange(0,553,25),552]):
        mask = cv2.erode(masks['object'][f].astype(np.uint8), np.ones((9,9),np.uint8)) > 0
        mask &= ~masks['person'][f]
        y,x = np.where(mask); selection = np.linspace(0,len(x)-1,min(400,len(x))).astype(int); x=x[selection];y=y[selection]
        raw = cv2.imdecode(depth['raw/front_stereo_camera_left'][f],-1)[y,x]/1000.
        aligned = cv2.imdecode(depth['aligned/front_stereo_camera_left'][f],-1)[y,x]/1000.
        zb = front_depth(bc[f],x,y); zp = front_depth(pc[f],x,y)
        evidence['sampled_depth'].append({'frame': int(f), 'raw_surface_m': float(np.median(raw)),
            'aligned_surface_m': float(np.median(aligned)), 'baseline_rendered_surface_m': float(np.nanmedian(zb)),
            'refined_rendered_surface_m': float(np.nanmedian(zp)),
            'aligned_minus_baseline_m': float(np.nanmedian(aligned-zb)),
            'aligned_minus_refined_m': float(np.nanmedian(aligned-zp))})
    V = np.load(root/'frame0_reference_human.npz')['vertices_world']
    V = (V-W[:3,3])@W[:3,:3]; uv=V@K.T;uv=uv[:,:2]/uv[:,2:]
    y,x=np.where(masks['person'][0])
    evidence['reference_human_projected_bbox_xyxy'] = [*uv.min(0).tolist(),*uv.max(0).tolist()]
    evidence['person_mask_bbox_xyxy'] = [int(x.min()),int(y.min()),int(x.max()),int(y.max())]
for key in ['aligned_minus_baseline_m','aligned_minus_refined_m']:
    vals=np.array([v[key] for v in evidence['sampled_depth']]);evidence[key]={'median':float(np.median(vals)), 'range':[float(vals.min()),float(vals.max())], 'std':float(vals.std())}
(out/'independent_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')

fig, ax = plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
frames=np.arange(len(p));ax[0].plot(frames,bc[:,2,3],color='#888888',lw=1,label='Baseline');ax[0].plot(frames,pc[:,2,3],color='#1763a6',lw=1.8,label='Refined');ax[0].set_ylabel('Camera depth (m)');ax[0].legend(loc='upper left');ax[0].grid(alpha=.2)
ax[1].plot(frames,(pc[:,2,3]-bc[:,2,3])*100,color='#1763a6',lw=1);ax[1].axhline(0,color='black',lw=.5);ax[1].set_ylabel('Depth correction (cm)');ax[1].set_xlabel('Frame');ax[1].grid(alpha=.2)
for a in ax:
 for start,end in json.loads((out/'diagnostics.json').read_text())['static_segments_start_end_exclusive']:
  a.axvspan(start,end-1,color='#52b58a',alpha=.12)
fig.savefig(out/'depth_comparison.png',dpi=160);plt.close(fig)
cap=cv2.VideoCapture(str(root/'video.mp4')); vertices=np.asarray(mesh.convex_hull.vertices); tiles=[]
for f in [0,100,250,350,500,552]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
    for T,color in [(bc[f],(50,50,230)),(pc[f],(40,190,40))]:
        xyz=vertices@T[:3,:3].T+T[:3,3];uv=xyz@K.T;uv=uv[:,:2]/uv[:,2:];poly=cv2.convexHull(uv.astype(np.float32));cv2.polylines(im,[np.rint(poly).astype(np.int32)],True,color,1,cv2.LINE_AA)
    crop=cv2.resize(im[550:660,695:815],(480,440),interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop,f'Frame {f}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2);tiles.append(crop)
cap.release()
cv2.imwrite(str(out/'projection_comparison.jpg'),np.concatenate([np.concatenate(tiles[:3],axis=1),np.concatenate(tiles[3:],axis=1)],axis=0))
print(json.dumps({k:v for k,v in evidence.items() if k!='sampled_depth'},indent=2))
