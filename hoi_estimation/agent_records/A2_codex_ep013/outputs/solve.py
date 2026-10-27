#!/usr/bin/env python3
"""Conservative, offline textured-mesh pose correction. See REPORT.md.
All scene-specific choices are documented next to the constants below.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time,random
from pathlib import Path
import cv2,h5py,numpy as np,torch,trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation, Slerp

# Inspection of video and mask corners shows the box at rest here. The person
# passing in front changes its visible mask, but not its pose. No motion or
# contact prior is imposed during carrying/pushing. Endpoints are conservative.
REST_INTERVALS=[(332,388),(486,692),(834,922)]
SIZE=500
STRIDE=3

class Renderer:
 def __init__(self,path):
  self.ctx=dr.RasterizeCudaContext()
  self.mesh=trimesh.load(path,force='scene',process=False).to_geometry()
  self.vertices=np.asarray(self.mesh.vertices)
  self.v=torch.tensor(self.vertices,dtype=torch.float32,device='cuda')
  self.f=torch.tensor(self.mesh.faces.astype(np.int32),device='cuda')
  self.uv=torch.tensor(self.mesh.visual.uv,dtype=torch.float32,device='cuda')[None]
  texture=np.asarray(self.mesh.visual.material.baseColorTexture.convert('RGB'))
  self.texture=torch.tensor(texture[::-1].copy()/255,dtype=torch.float32,device='cuda')[None]
 def render(self,T,K):
  with torch.no_grad():
   T=torch.as_tensor(T,dtype=torch.float32,device='cuda');K=torch.as_tensor(K,dtype=torch.float32,device='cuda')
   v=self.v@T[:3,:3].T+T[:3,3];z=v[:,2]
   clip=torch.stack((2*(K[0,0]*v[:,0]+K[0,2]*z)/SIZE-z,2*(K[1,1]*v[:,1]+K[1,2]*z)/SIZE-z,10.001/9.999*z-.020002,z),-1)[None]
   rast,_=dr.rasterize(self.ctx,clip,self.f,resolution=[SIZE,SIZE])
   uv,_=dr.interpolate(self.uv,rast,self.f)
   color=dr.texture(self.texture,uv,filter_mode='linear')
   xyz,_=dr.interpolate(self.v[None],rast,self.f)
   mask=(rast[0,:,:,3]>0).cpu().numpy()
   return color[0].cpu().numpy(),xyz[0].cpu().numpy(),mask

def clean_mask(mask):
 n,lab,st,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
 if n<=1:return mask.astype('uint8')
 areas=st[1:,cv2.CC_STAT_AREA];keep=np.where(areas>=max(12,areas.max()*.05))[0]+1
 return np.isin(lab,keep).astype('uint8')

def prepare(im,mask,person,T,K,vertices):
 mask=clean_mask(mask);ys,xs=np.where(mask)
 v=vertices[::40]@T[:3,:3].T+T[:3,3];uv=v@K.T;uv=uv[:,:2]/uv[:,2:]
 lo=uv.min(0);hi=uv.max(0)
 if len(xs):lo=np.minimum(lo,[xs.min(),ys.min()]);hi=np.maximum(hi,[xs.max(),ys.max()])
 center=(lo+hi)/2;extent=max(hi-lo)*1.6;scale=SIZE/extent
 M=np.array([[scale,0,-(center[0]-extent/2)*scale],[0,scale,-(center[1]-extent/2)*scale]])
 k=K.copy();k[:2,2]-=center-extent/2;k[:2]*=scale
 rgb=cv2.warpAffine(im,M,(SIZE,SIZE))[:,:,::-1].copy()
 mask=cv2.warpAffine(mask,M,(SIZE,SIZE),flags=cv2.INTER_NEAREST)>0
 person=cv2.warpAffine(person.astype('uint8'),M,(SIZE,SIZE),flags=cv2.INTER_NEAREST)>0
 return rgb,mask,person,k,scale

def metrics(color,pred,rgb,mask,person):
 visible=pred&(~person|mask)
 iou=(visible&mask).sum()/max(1,(visible|mask).sum())
 valid=cv2.erode((pred&mask).astype('uint8'),np.ones((7,7),np.uint8))>0
 if valid.sum()<100:return iou,1.
 a=cv2.GaussianBlur(color,(0,0),2)[valid]
 b=cv2.GaussianBlur(rgb.astype('float32')/255,(0,0),2)[valid]
 a=a/max(.1,np.quantile(a.mean(1),.75));b=b/max(.1,np.quantile(b.mean(1),.75))
 photo=np.minimum(((a-b)**2).mean(1),.2).mean()
 return float(iou),float(photo)

def fit_frame(index,rgb,mask,person,K,base,renderer,sift,reference_rotations):
 # Re-seeding for each frame makes results independent of frame processing order.
 cv2.setRNGSeed(417+index)
 kp,desc=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.dilate(mask.astype('uint8'),np.ones((9,9),np.uint8)))
 if desc is None:return None,{'frame':index,'reason':'no_features'}
 col,_,ms=renderer.render(base,K);base_iou,base_photo=metrics(col,ms,rgb,mask,person)
 candidates=[]
 for d in [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]:
  T=base.copy();T[:3,:3]=base[:3,:3]@np.diag(d);candidates.append(T)
 # After the second lift the baseline changes the dimension that is vertical.
 # Coarse views from the earlier correctly identified desk/floor orientations
 # provide independent 2D--3D matches; no pose is accepted just from this prior.
 if index>=750:
  for rot in reference_rotations:
   T=base.copy();T[:3,:3]=rot;candidates.append(T)
 best=None;best_score=-1e9;info={'frame':index,'baseline_iou':base_iou,'baseline_photo':base_photo}
 matcher=cv2.BFMatcher()
 for ci,initial in enumerate(candidates):
  col,xyz,ms=renderer.render(initial,K)
  image=(np.clip(col*255,0,255).astype('uint8'));image[~ms]=180
  image=cv2.GaussianBlur(image,(0,0),1.4)
  ka,da=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),cv2.erode(ms.astype('uint8'),np.ones((9,9),np.uint8)))
  if da is None:continue
  pairs=matcher.knnMatch(da,desc,k=2)
  matches=[a for a,b in pairs if a.distance<.8*b.distance]
  # Repeated text must not yield many votes for the same observed keypoint.
  unique={}
  for m in matches:
   if m.trainIdx not in unique or m.distance<unique[m.trainIdx].distance:unique[m.trainIdx]=m
  matches=list(unique.values())
  if len(matches)<12:continue
  obj=np.array([xyz[int(round(ka[m.queryIdx].pt[1])),int(round(ka[m.queryIdx].pt[0]))] for m in matches],np.float64)
  img=np.array([kp[m.trainIdx].pt for m in matches],np.float64)
  ok,rv,tv,ii=cv2.solvePnPRansac(obj,img,K,None,iterationsCount=2000,reprojectionError=4.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
  if not ok or ii is None or len(ii)<12:continue
  ids=ii[:,0]
  # EPNP sometimes returns a bad solution for almost planar, repeated text.
  if not np.isfinite(tv).all() or not 1.5<float(tv[2,0])<5:continue
  try:cv2.solvePnPRefineLM(obj[ids],img[ids],K,None,rv,tv)
  except cv2.error:continue
  T=np.eye(4);T[:3,:3]=cv2.Rodrigues(rv)[0];T[:3,3]=tv[:,0]
  if not np.isfinite(T).all() or np.linalg.norm(T[:3,3]-base[:3,3])>.65:continue
  errors=np.linalg.norm(cv2.projectPoints(obj,rv,tv,K,None)[0][:,0]-img,axis=1)
  ids=np.where(errors<4)[0]
  if len(ids)<12:continue
  # Require correspondences distributed across the object, not a tiny logo.
  spread=np.sqrt(np.linalg.det(np.cov(img[ids].T)))
  if spread<140:continue
  col,_,ms=renderer.render(T,K);iou,photo=metrics(col,ms,rgb,mask,person)
  angle=np.degrees(Rotation.from_matrix(T[:3,:3]@base[:3,:3].T).magnitude())
  initial_angle=np.degrees(Rotation.from_matrix(T[:3,:3]@initial[:3,:3].T).magnitude())
  improvement=(base_photo-photo)/max(base_photo,.005)
  # Large orientation changes need both texture and silhouette evidence.
  if iou<.80 or iou<base_iou-.045 or photo>.115:continue
  if angle>45:
   if len(ids)<18 or improvement<.12:continue
  else:
   # Preserve strong baseline poses unless the outline improves, or dense
   # texture independently supports a change in metric depth/orientation.
   if not (iou>base_iou+.025 or (len(ids)>=24 and improvement>.18 and iou>base_iou-.015)):continue
  # A repeated print on another face may pass RANSAC, but has worse texture.
  score=len(ids)*.12+iou*12-photo*100
  if score>best_score:
   best_score=score;best=T
   info.update(accepted=True,inliers=len(ids),iou=iou,photo=photo,photo_improvement=improvement,angle_change_deg=angle,proposal=ci,spread=float(spread),median_reprojection=float(np.median(errors[ids])),score=score)
 return best,info

def robust_pose(poses):
 ts=poses[:,:3,3];center=np.median(ts,axis=0);rs=Rotation.from_matrix(poses[:,:3,:3]);mean=rs.mean()
 dist=np.linalg.norm(ts-center,axis=1);ang=(mean.inv()*rs).magnitude()
 keep=(dist<max(.025,np.median(dist)*2.5))&(ang<max(np.deg2rad(4),np.median(ang)*2.5))
 if keep.sum()<3:keep=np.ones(len(poses),bool)
 T=np.eye(4);T[:3,3]=np.median(ts[keep],axis=0);T[:3,:3]=Rotation.from_matrix(poses[keep,:3,:3]).mean().as_matrix()
 return T,keep

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
 start=time.perf_counter();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);cv2.setNumThreads(1)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 inputs=Path(args.inputs);out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 r=np.load(inputs/'reconstruction.npz');baseline=r['T_world_mesh'].astype(np.float64);F=len(baseline)
 camera=json.load(open(inputs/'camera.json'));K=np.array(camera['K'],np.float64);C=np.array(camera['T_world_cam'],np.float64);base=np.linalg.inv(C)[None]@baseline
 renderer=Renderer(str(inputs/'object_mesh.glb'))
 sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.008,edgeThreshold=15,sigma=1.2)
 references=[base[0,:3,:3],base[min(480,F-1),:3,:3]@np.diag([-1,-1,1])]
 estimates={};infos=[];cap=cv2.VideoCapture(str(inputs/'video.mp4'))
 selected=set(range(0,F,STRIDE))|{F-1}
 # Read HDF5 in temporal chunk order: individual random slices repeatedly
 # decompress the entire 29-frame chunk and are substantially slower.
 with h5py.File(inputs/'masks.h5','r') as h:
  batch=h['object'].chunks[0] if h['object'].chunks else 16
  for lo in range(0,F,batch):
   hi=min(F,lo+batch);objects=h['object'][lo:hi];people=h['person'][lo:hi]
   for index in range(lo,hi):
    ok,im=cap.read()
    if not ok:raise RuntimeError('Missing video frame '+str(index))
    if index not in selected:continue
    rgb,mask,person,k,scale=prepare(im,objects[index-lo],people[index-lo],base[index],K,renderer.vertices)
    T,info=fit_frame(index,rgb,mask,person,k,base[index],renderer,sift,references)
    infos.append(info)
    if T is not None:estimates[index]=T
   if lo% (batch*3)==0:print(f'frames {hi}/{F}; accepted {len(estimates)}; elapsed {time.perf_counter()-start:.1f}s',flush=True)
 cap.release()
 result=base.copy();source=np.zeros(F,np.int32)
 # Only bridge short (<=0.3s) gaps supported by nearby successful fits.
 ids=np.array(sorted(estimates),int)
 if len(ids):
  for i in ids:result[i]=estimates[i];source[i]=1
  for a,b in zip(ids[:-1],ids[1:]):
   if b-a>9:continue
   aa=estimates[a];bb=estimates[b]
   if np.linalg.norm(aa[:3,3]-bb[:3,3])>.25:continue
   if (Rotation.from_matrix(aa[:3,:3]).inv()*Rotation.from_matrix(bb[:3,:3])).magnitude()>np.deg2rad(35):continue
   # Do not interpolate over a baseline rotation reset at an occluded frame.
   rr=Slerp([a,b],Rotation.from_matrix(np.stack([aa[:3,:3],bb[:3,:3]])))
   for i in range(a+1,b):
    u=(i-a)/(b-a);result[i,:3,3]=(1-u)*aa[:3,3]+u*bb[:3,3];result[i,:3,:3]=rr([i]).as_matrix()[0];source[i]=2
  # A texture-confirmed half-turn at both ends of a short occlusion remains
  # valid while the baseline rotation evolves continuously (e.g. a foot push).
  # Propagate only this discrete identity correction, retaining baseline
  # translation and local rotation throughout the gap.
  half_turn=np.diag([-1.,-1.,1.]);rz=Rotation.from_matrix(half_turn)
  for a,b in zip(ids[:-1],ids[1:]):
   if not 9<b-a<=60:continue
   ca=Rotation.from_matrix(base[a,:3,:3].T@estimates[a][:3,:3]);cb=Rotation.from_matrix(base[b,:3,:3].T@estimates[b][:3,:3])
   if max((rz.inv()*ca).magnitude(),(rz.inv()*cb).magnitude())>np.deg2rad(12):continue
   steps=(Rotation.from_matrix(base[a:b,:3,:3]).inv()*Rotation.from_matrix(base[a+1:b+1,:3,:3])).magnitude()
   if steps.max()>np.deg2rad(10):continue
   for i in range(a+1,b):
    if source[i]==0:result[i,:3,:3]=base[i,:3,:3]@half_turn;source[i]=5
 rest_reports=[]
 # The box also stays fixed on the initial desk through frame 139. The
 # baseline's depth drifts by ~17 cm although its image corners stay fixed.
 # Select among existing poses using the unoccluded frame-0 image, rather
 # than introduce a new monocular scale or trust a planar PnP fit here.
 initial_cap=cv2.VideoCapture(str(inputs/'video.mp4'));ok,initial_image=initial_cap.read();initial_cap.release()
 if ok and F>=140:
  with h5py.File(inputs/'masks.h5','r') as initial_masks:
   rgb,mask,person,k,_=prepare(initial_image,initial_masks['object'][0],initial_masks['person'][0],base[0],K,renderer.vertices)
  initial_scores=[]
  for i in range(140):
   col,_,ms=renderer.render(base[i],k);iou,photo=metrics(col,ms,rgb,mask,person)
   initial_scores.append((iou-.5*photo,i,iou,photo))
  initial_scores.sort(reverse=True)
  top=initial_scores[:5]
  pose,_=robust_pose(np.stack([base[x[1]] for x in top]))
  col,_,ms=renderer.render(pose,k);iou,photo=metrics(col,ms,rgb,mask,person)
  col0,_,ms0=renderer.render(base[0],k);iou0,photo0=metrics(col0,ms0,rgb,mask,person)
  if iou>=iou0 and photo<=photo0+.015:
   result[:140]=pose;source[:140]=4
   rest_reports.append(dict(start=0,end=140,method='existing_pose_selection_on_unoccluded_frame0',candidate_frames=[x[1] for x in top],iou=iou,photo=photo))
 for a,b in REST_INTERVALS:
  b=min(b,F);valid=[i for i in ids if a<=i<b]
  if len(valid)<5:continue
  ps=np.stack([estimates[i] for i in valid]);pose,keep=robust_pose(ps)
  # Pool only when accepted image fits agree that this interval is stationary.
  scatter=np.median(np.linalg.norm(ps[:,:3,3]-pose[:3,3],axis=1));angular=np.median((Rotation.from_matrix(pose[:3,:3]).inv()*Rotation.from_matrix(ps[:,:3,:3])).magnitude())
  if scatter>.035 or angular>np.deg2rad(5):continue
  result[a:b]=pose;source[a:b]=3
  rest_reports.append(dict(start=a,end=b,accepted_fits=len(valid),retained_fits=int(keep.sum()),translation_scatter_m=float(scatter),angular_scatter_deg=float(np.degrees(angular))))
 world=C[None]@result
 # The box is visibly in contact with the carpet throughout [306,700).
 # PnP alone inherits focal-length uncertainty and can leave it floating.
 # Intersect the object's center ray with the known support plane, accounting
 # for its rotated mesh extent. This preserves the projected center and uses
 # released world geometry to anchor depth. The provided reference human's
 # lowest point is only 9 mm below this plane, supporting this contact prior.
 plane=np.array(json.load(open(inputs/'ground_plane.json'))['plane'],np.float64)
 support_vertices=renderer.vertices[::10]
 contact_adjustments=[]
 for i in range(306,min(700,F)):
  if source[i]==0:continue
  pts=support_vertices@world[i,:3,:3].T+world[i,:3,3]
  distance=float(np.min(pts@plane[:3]+plane[3]))
  ray=world[i,:3,3]-C[:3,3];denom=float(ray@plane[:3])
  if abs(denom)<.1:continue
  delta=-distance/denom*ray
  if np.linalg.norm(delta)>.35:continue
  world[i,:3,3]+=delta
  contact_adjustments.append(float(np.linalg.norm(delta)))
 # Preserve original bit patterns (apart from the required float64 cast) for
 # every frame without an accepted correction.
 world[source==0]=baseline[source==0]
 world[:,3,:]=[0,0,0,1]
 assert world.shape==(F,4,4) and np.isfinite(world).all()
 assert np.max(np.abs(np.linalg.det(world[:,:3,:3])-1))<1e-4
 np.savez_compressed(out,T_world_mesh=world.astype(np.float64))
 elapsed=time.perf_counter()-start
 audit=dict(frames=F,elapsed_seconds=elapsed,keyframes_tested=len(infos),keyframes_accepted=len(estimates),modified_frames=int((source>0).sum()),unchanged_frames=int((source==0).sum()),source_counts={str(i):int((source==i).sum()) for i in range(6)},rest_intervals=rest_reports,keyframes=infos)
 audit['floor_contact']=dict(start=306,end=min(700,F),adjusted_frames=len(contact_adjustments),median_adjustment_m=float(np.median(contact_adjustments)) if contact_adjustments else 0.)
 with open(out.parent/'diagnostics.json','w') as f:json.dump(audit,f,indent=2)
 np.savez_compressed(out.parent/'fit_diagnostics.npz',source=source,keyframes=ids,T_cam_mesh=np.stack([estimates[i] for i in ids]) if len(ids) else np.empty((0,4,4)))
 print(json.dumps({k:v for k,v in audit.items() if k not in ['keyframes']},indent=2),flush=True)
if __name__=='__main__':main()
