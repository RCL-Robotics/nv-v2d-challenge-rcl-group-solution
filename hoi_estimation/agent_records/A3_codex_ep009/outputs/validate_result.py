"""Independent artifact checks and a raw-mask comparison/visual inspection sheet."""
import hashlib,json,cv2,h5py,numpy as np,torch
from pathlib import Path
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from solve import SilhouetteRenderer,SIDE
import trimesh

cv2.setNumThreads(1);torch.set_num_threads(4)
p=Path('/work/inputs');out=Path('/work/outputs');diag=out/'diagnostics'
base=np.load(p/'reconstruction.npz')['T_world_mesh'].astype(np.float64)
pose=np.load(out/'object_poses_world.npz')['T_world_mesh']
first=np.load(diag/'first_run.npz')['T_world_mesh']
assert pose.shape==(870,4,4) and pose.dtype==np.float64
assert np.array_equal(pose,first),f'Non-repeatable result: {np.abs(pose-first).max()}'
assert np.isfinite(pose).all()
c=json.loads((p/'camera.json').read_text());K=np.array(c['K']);C=np.array(c['T_world_cam']);bc=np.linalg.inv(C)@base;pc=np.linalg.inv(C)@pose
tr=np.linalg.norm(pose[:,:3,3]-base[:,:3,3],axis=1);changed=tr>1e-10
rots=np.rad2deg(Rotation.from_matrix(base[:,:3,:3].transpose(0,2,1)@pose[:,:3,:3]).magnitude())
mesh=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry();renderer=SilhouetteRenderer(mesh)
proj=bc[:,:3,3]@K.T;orig=np.floor(proj[:,:2]/proj[:,2:]-SIDE/2).astype(int);Ks=np.repeat(K[None],len(pc),0);Ks[:,:2,2]-=orig
bm=renderer.render_all(bc,Ks);pm=renderer.render_all(pc,Ks)
raw_iou=[]
with h5py.File(p/'masks.h5') as h:
 for i in np.where(changed)[0]:
  x,y=orig[i];obs=h['object'][i,y:y+SIDE,x:x+SIDE]
  def iou(a):return float(((a>.5)&obs).sum()/((a>.5)|obs).sum())
  raw_iou.append([iou(bm[i]),iou(pm[i])])
raw_iou=np.array(raw_iou)
frames=[0,265,299,312,350,383,400,425,600,710,740,820]
w=288;sheet=Image.new('RGB',(4*w,3*w),'white');d=ImageDraw.Draw(sheet);cap=cv2.VideoCapture(str(p/'video.mp4'))
for j,i in enumerate(frames):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read();assert ok
 x,y=orig[i];im=cv2.cvtColor(bgr[y:y+SIDE,x:x+SIDE],cv2.COLOR_BGR2RGB)
 for m,col in [(bm[i],(0,255,255)),(pm[i],(255,0,255))]:
  contours,_=cv2.findContours((m>.5).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,col,1)
 im=cv2.resize(im,(w,w));x0,y0=j%4*w,j//4*w;sheet.paste(Image.fromarray(im),(x0,y0))
 d.text((x0+5,y0+5),f'Frame {i}; translation {tr[i]*100:.2f} cm; heading {rots[i]:.1f} deg',fill='white',stroke_fill='black',stroke_width=1)
sheet.save(diag/'final_overlay.jpg',quality=95)
old_steps=np.rad2deg(Rotation.from_matrix(base[:-1,:3,:3].transpose(0,2,1)@base[1:,:3,:3]).magnitude())
new_steps=np.rad2deg(Rotation.from_matrix(pose[:-1,:3,:3].transpose(0,2,1)@pose[1:,:3,:3]).magnitude())
checks={'repeatability_bitwise_equal':True,'output_array_sha256':hashlib.sha256(pose.tobytes()).hexdigest(),
 'raw_mask_iou_before_mean':raw_iou[:,0].mean(),'raw_mask_iou_after_mean':raw_iou[:,1].mean(),
 'raw_mask_iou_improved_frames':int(sum(raw_iou[:,1]>raw_iou[:,0])),
 'raw_mask_iou_compared_frames':len(raw_iou),
 'rotation_corrections_degrees':{str(i):float(rots[i]) for i in [265,266,383]},
 'heading_spike_frame_steps':{str(i):{'before':old_steps[i-1:i+1].tolist(),'after':new_steps[i-1:i+1].tolist()} for i in [265,266,383]},
 'input_sha256':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['camera.json','reconstruction.npz','object_mesh.glb','video.mp4','masks.h5']}}
(diag/'independent_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
