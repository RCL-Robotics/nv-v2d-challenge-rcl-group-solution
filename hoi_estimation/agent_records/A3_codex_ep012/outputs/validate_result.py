"""Independent file/invariant checks and video-edge sensitivity evaluation."""
import json
from pathlib import Path
import cv2,h5py,numpy as np,torch,trimesh,nvdiffrast.torch as dr
from PIL import Image,ImageDraw
p=Path('/work/inputs'); out=Path('/work/outputs'); c=json.loads((p/'camera.json').read_text()); K=np.array(c['K']); W=np.array(c['T_world_cam'])
b=np.load(p/'reconstruction.npz')['T_world_mesh'];r=np.load(out/'object_poses_world.npz')['T_world_mesh'];n=len(b)
assert r.shape==(700,4,4) and r.dtype==np.float64 and np.isfinite(r).all()
assert np.max(np.abs(r[:,3]-[0,0,0,1]))<1e-10
assert np.max(np.abs(np.linalg.det(r[:,:3,:3])-1))<3e-5
assert np.max(np.abs(r[:,:3,:3].transpose(0,2,1)@r[:,:3,:3]-np.eye(3)))<3e-5
bc=np.linalg.inv(W)[None]@b;rc=np.linalg.inv(W)[None]@r
assert np.max(np.abs(np.linalg.norm(bc[:,:3,3],axis=1)-np.linalg.norm(rc[:,:3,3],axis=1)))<1e-12
mesh=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry();ctx=dr.RasterizeCudaContext();v=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');tri=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');kt=torch.tensor(K.T,dtype=torch.float32,device='cuda')
uv=bc[:,:3,3]@K.T;uv=uv[:,:2]/uv[:,2:];origin=np.rint(uv-128).astype(int)
thresholds=[(15,40),(25,65),(40,100)]
scores=np.zeros((n,2,len(thresholds)));cap=cv2.VideoCapture(str(p/'video.mp4'));m=h5py.File(p/'masks.h5','r');sample=[0,70,140,210,245,280,315,350,385,420,455,490,525,560,595,630,699];sheet=Image.new('RGB',(768,256*len(sample)));sample_i=0
for first in range(0,n,20):
 end=min(n,first+20);outlines=[]
 for poses in [bc,rc]:
  with torch.no_grad():
   T=torch.tensor(poses[first:end],dtype=torch.float32,device='cuda');vc=v[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];proj=vc@kt;o=torch.tensor(origin[first:end],device='cuda');px=proj[:,:,0]/proj[:,:,2]-o[:,None,0];py=proj[:,:,1]/proj[:,:,2]-o[:,None,1];z=vc[:,:,2];clip=torch.stack([(2*px/256-1)*z,(2*py/256-1)*z,((z-.1)/9.9*2-1)*z,z],-1);rast,_=dr.rasterize(ctx,clip,tri,[256,256]);alpha=(rast[:,:,:,3]>0).cpu().numpy()
  contours=[cv2.findContours(a.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0] for a in alpha];outlines.append(contours)
 for i,f in enumerate(range(first,end)):
  _,bgr=cap.read();x,y=origin[f];im=cv2.cvtColor(bgr[y:y+256,x:x+256],cv2.COLOR_BGR2RGB);gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);person=m['person'][f,y:y+256,x:x+256].astype(np.uint8);valid=1-cv2.dilate(person,np.ones((5,5),np.uint8))
  for j,cs in enumerate(outlines):
   outline=np.zeros((256,256),np.uint8);cv2.drawContours(outline,cs[i],-1,1,1);keep=(outline>0)&(valid>0)
   for k,(lo,hi) in enumerate(thresholds):
    edge=cv2.Canny(cv2.GaussianBlur(gray,(3,3),.6),lo,hi);dist=cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);scores[f,j,k]=np.minimum(dist[keep],5).mean()
  if f in sample:
   mask=m['object'][f,y:y+256,x:x+256].astype(np.uint8);mc=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
   for j in range(3):
    cell=im.copy()
    if j:
     cv2.drawContours(cell,mc,-1,(0,255,80),1);cv2.drawContours(cell,outlines[j-1][i],-1,(255,70,30),1)
    cell=Image.fromarray(cell);ImageDraw.Draw(cell).text((4,4),f'{f}: '+['video','baseline','result'][j],fill='yellow');sheet.paste(cell,(j*256,sample_i*256))
   sample_i+=1
cap.release();m.close();sheet.save(out/'final_comparison.jpg')
vs=np.array(mesh.vertices)[::100];displacements=[]
for i in range(n):
 old=vs@b[i,:3,:3].T+b[i,:3,3];new=vs@r[i,:3,:3].T+r[i,:3,3];displacements.append(np.linalg.norm(new-old,axis=1).mean())
summary={'shape':list(r.shape),'dtype':str(r.dtype),'finite':True,'max_rotation_orthogonality_error':float(np.max(np.abs(r[:,:3,:3].transpose(0,2,1)@r[:,:3,:3]-np.eye(3)))),'mean_mesh_displacement_from_baseline_m':float(np.mean(displacements)),'max_frame_mean_mesh_displacement_from_baseline_m':float(np.max(displacements)), 'edge_sensitivity':{str(t):{'baseline_pixels':float(scores[:,0,k].mean()),'result_pixels':float(scores[:,1,k].mean())} for k,t in enumerate(thresholds)}}
(out/'independent_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
