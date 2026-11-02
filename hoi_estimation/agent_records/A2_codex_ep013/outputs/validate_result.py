"""Output validation and qualitative evidence, independent of pose selection."""
from pathlib import Path
import json,cv2,numpy as np,h5py
from solve import Renderer,prepare,metrics,REST_INTERVALS
p=Path('/work/inputs');o=Path('/work/outputs')
base=np.load(p/'reconstruction.npz')['T_world_mesh'].astype('float64');new=np.load(o/'object_poses_world.npz')['T_world_mesh'];audit=np.load(o/'fit_diagnostics.npz');src=audit['source'];c=json.load(open(p/'camera.json'));C=np.array(c['T_world_cam']);K=np.array(c['K']);b=np.linalg.inv(C)[None]@base;n=np.linalg.inv(C)[None]@new
assert new.shape==(922,4,4) and new.dtype==np.float64 and np.isfinite(new).all()
assert np.array_equal(new[src==0],base[src==0])
assert np.array_equal(new[:,3],np.tile([0.,0.,0.,1.],(922,1)))
orth=np.max(np.abs(new[:,:3,:3].transpose(0,2,1)@new[:,:3,:3]-np.eye(3)))
assert orth<2e-5 # Preserved float32 baseline rotations reach 1.23e-5.
renderer=Renderer(str(p/'object_mesh.glb'));vc=cv2.VideoCapture(str(p/'video.mp4'));rows=[];records=[]
with h5py.File(p/'masks.h5') as h:
 for i in [0,180,300,360,420,480,660,720,780,840,921]:
  vc.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=vc.read();mask=h['object'][i];person=h['person'][i];rgb,mask,person,k,scale=prepare(im,mask,person,b[i],K,renderer.vertices)
  col,_,ms=renderer.render(b[i],k);bi,bp=metrics(col,ms,rgb,mask,person)
  col2,_,ms2=renderer.render(n[i],k);ni,npht=metrics(col2,ms2,rgb,mask,person)
  rec=dict(frame=i,source=int(src[i]),baseline_iou=bi,result_iou=ni,baseline_photo=bp,result_photo=npht);records.append(rec)
  col[~ms]=.7;col2[~ms2]=.7
  images=[rgb[:,:,::-1].copy(),(np.clip(col*255,0,255).astype('uint8'))[:,:,::-1].copy(),(np.clip(col2*255,0,255).astype('uint8'))[:,:,::-1].copy()]
  for j,(image,title) in enumerate(zip(images,[f'Video {i}',f'Baseline IoU={bi:.2f}',f'Result IoU={ni:.2f}'])):
   cv2.putText(image,title,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,30,240),2)
  rows.append(cv2.resize(np.hstack(images),(1050,350)))
cv2.imwrite(str(o/'result_comparison.jpg'),np.vstack(rows))
plane=np.array(json.load(open(p/'ground_plane.json'))['plane']);vv=renderer.vertices[::30];floor={}
for name,traj in [('baseline',base),('result',new)]:
 ds=[]
 for T in traj[486:692]:ds.append(np.min((vv@T[:3,:3].T+T[:3,3])@plane[:3]+plane[3]))
 floor[name]={'median_lowest_point_plane_distance_m':float(np.median(ds)),'translation_axis_std_m':np.std(traj[486:692,:3,3],axis=0).tolist()}
report=dict(shape=list(new.shape),dtype=str(new.dtype),finite=bool(np.isfinite(new).all()),max_rotation_orthogonality_error=float(orth),max_rotation_determinant_error=float(np.max(np.abs(np.linalg.det(new[:,:3,:3])-1))),untouched_frames_exact=bool(np.array_equal(new[src==0],base[src==0])),examples=records,stationary_floor=floor)
json.dump(report,open(o/'validation.json','w'),indent=2);print(json.dumps(report,indent=2))
