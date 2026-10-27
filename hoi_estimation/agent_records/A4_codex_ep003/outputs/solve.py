#!/usr/bin/env python3
"""Offline image-driven mug pose repair.

All image fitting, temporal selection, and output generation are recomputed.
Uses only released inputs and installed numerical/rendering packages.
The apparent image scale is a nuisance for matching, NEVER a pose/depth update.
"""
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np, torch, trimesh
import nvdiffrast.torch as dr
class Renderer:
 def __init__(self,path,K,size=128):
  self.device='cuda';self.size=size;self.K=np.array(K);m=trimesh.load(path,force='scene',process=False).to_geometry();self.mesh=m
  self.v=torch.tensor(np.asarray(m.vertices),device='cuda',dtype=torch.float32);self.faces=torch.tensor(np.array(m.faces,dtype=np.int32),device='cuda');self.uv=torch.tensor(np.array(m.visual.uv),device='cuda',dtype=torch.float32)[None]
  mat=m.visual.material; im=getattr(mat,'baseColorTexture',None)
  if im is None: im=mat.image
  self.tex=torch.tensor(np.array(im.convert('RGB'))[::-1].copy(),device='cuda',dtype=torch.float32)[None]/255
  self.ctx=dr.RasterizeCudaContext()
 def render(self,R,t,origin,aa=True,scale=None):
  if not torch.is_tensor(R): R=torch.tensor(R,device='cuda',dtype=torch.float32)
  if not torch.is_tensor(t): t=torch.tensor(t,device='cuda',dtype=torch.float32)
  if not torch.is_tensor(origin): origin=torch.tensor(origin,device='cuda',dtype=torch.float32)
  if R.ndim==2:R=R[None]
  if t.ndim==1:t=t[None]
  if origin.ndim==1:origin=origin[None]
  v=self.v[None]@R.transpose(1,2)+t[:,None];z=v[:,:,2];x=self.K[0,0]*v[:,:,0]+(self.K[0,2]-origin[:,0:1])*z;y=self.K[1,1]*v[:,:,1]+(self.K[1,2]-origin[:,1:2])*z
  if scale is not None:
   cx=(self.K[0,0]*t[:,0]/t[:,2]+self.K[0,2]-origin[:,0])[:,None]*z;cy=(self.K[1,1]*t[:,1]/t[:,2]+self.K[1,2]-origin[:,1])[:,None]*z
   x=cx+scale[:,None]*(x-cx);y=cy+scale[:,None]*(y-cy)
  clip=torch.stack([2*x/self.size-z,2*y/self.size-z,(10.01/9.99)*z-0.2002/9.99,z],-1)
  rast,_=dr.rasterize(self.ctx,clip,self.faces,resolution=[self.size,self.size]);uv,_=dr.interpolate(self.uv,rast,self.faces);color=dr.texture(self.tex,uv,filter_mode='linear');mask=(rast[:,:,:,3:]>0).float();color=color*mask
  if aa:mask=dr.antialias(mask,rast,clip,self.faces);color=dr.antialias(color,rast,clip,self.faces)
  return color,mask

import os,time,json,argparse,random
import numpy as np,cv2,h5py,torch
from scipy.spatial.transform import Rotation

def data(inputs,step=1):
 cam=json.load(open(inputs+'/camera.json'));K=np.array(cam['K']);W=np.array(cam['T_world_cam']);base=np.load(inputs+'/reconstruction.npz')['T_world_mesh'].astype(np.float64);T=np.linalg.inv(W)@base;F=len(T);S=128
 uv=T[:,:3,3]@K.T;uv=uv[:,:2]/uv[:,2:];orig=np.rint(uv-S/2).astype(int)
 images=[];masks=[];people=[]
 c=cv2.VideoCapture(inputs+'/video.mp4')
 with h5py.File(inputs+'/masks.h5') as h:
  for f in range(F):
   ok,im=c.read()
   if f%step:continue
   x,y=orig[f];images.append(im[y:y+S,x:x+S,::-1].copy()/255.);masks.append(h['object'][f,y:y+S,x:x+S]);people.append(h['person'][f,y:y+S,x:x+S])
 return cam,base,T,orig[::step],np.array(images),np.array(masks),np.array(people)

def rot6(x):
 a=torch.nn.functional.normalize(x[...,:3],dim=-1);b=x[...,3:];b=torch.nn.functional.normalize(b-a*(a*b).sum(-1,keepdim=True),dim=-1);return torch.stack([a,b,torch.cross(a,b,dim=-1)],-1)
def six(R):return np.concatenate([R[...,0],R[...,1]],-1)
def translation(x,t0,K):
 # Conserve the Euclidean camera-to-mesh-origin distance exactly.
 u=t0[:,0]/t0[:,2]+x[:,0]/K[0,0];v=t0[:,1]/t0[:,2]+x[:,1]/K[1,1];ray=torch.stack([u,v,torch.ones_like(u)],-1);return torch.nn.functional.normalize(ray,dim=-1)*t0.norm(dim=-1,keepdim=True)

def loss_fn(rgb,alpha,img,mask,person,dist):
 a=alpha[...,0];area=mask.sum((1,2)).clamp(min=1);visible=(1-person)+mask;visible=visible.clamp(max=1)
 sil=((a-mask).square()*visible).sum((1,2))/area
 outside=(a*dist*visible).sum((1,2))/area
 # Fit an exposure gain and offset, then compare the longitudinal silver/black pattern.
 weights=(a*mask).detach();den=weights.sum((1,2)).clamp(min=1);x=rgb.sum(-1)/3;y=img.sum(-1)/3
 mx=(x*weights).sum((1,2))/den;my=(y*weights).sum((1,2))/den
 dx=x-mx[:,None,None];dy=y-my[:,None,None]
 gain=((dx*dy*weights).sum((1,2))/((dx*dx*weights).sum((1,2))+1e-5)).clamp(.3,3).detach()
 err=(gain[:,None,None]*dx-dy).square();photo=(err*weights).sum((1,2))/den
 return sil+0.12*outside+12*photo,torch.stack([sil,outside,photo],-1)

def fit(inputs,out,step=4,iters=45):
 torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_num_threads(4)
 start=time.time();cam,base,T,orig,imgs,masks,people=data(inputs,step);idx=np.arange(0,len(T),step);K=np.array(cam['K']);ren=Renderer(inputs+'/object_mesh.glb',K);D='cuda'
 dist=np.array([np.minimum(cv2.distanceTransform((~m).astype('uint8'),cv2.DIST_L2,5),12)**2 for m in masks]);frames=[];scores=[];components=[];trans=[];scales=[]
 # Cover all handle azimuths. The video shows the lid above the base throughout;
 # the silhouette initialization enforces that observed polarity, including at baseline flips.
 local=np.array([Rotation.from_euler('x',flip,degrees=True).as_matrix()@Rotation.from_euler('z',angle,degrees=True).as_matrix() for flip in [0] for angle in range(0,360,30)])
 # A mug never inverts in this clip. The visible silhouette gives its long-axis
 # direction in the image; a 9-frame median suppresses isolated depth-axis flips
 # without estimating depth from object size. These are initializations only.
 from scipy.ndimage import median_filter
 zseed=median_filter(T[:,2,2],size=9,mode='nearest')
 initialR=T[:,:3,:3].copy()
 for n,f in enumerate(idx):
  yy,xx=np.where(masks[n]);points=np.stack([xx,yy],-1).astype(float)
  _,vectors=np.linalg.eigh(np.cov(points.T));axis=vectors[:,-1]
  if axis[1]<0:axis=-axis
  z=np.clip(zseed[f],-.85,.6);desired=np.r_[axis*np.sqrt(1-z*z),z]
  old=initialR[f,:,2];cross=np.cross(old,desired);sin=np.linalg.norm(cross);cos=np.dot(old,desired)
  if sin>1e-8: initialR[f]=Rotation.from_rotvec(cross/sin*np.arctan2(sin,cos)).as_matrix()@initialR[f]
 B=len(local)
 for n,f in enumerate(idx):
  init=initialR[f][None]@local;rr=torch.tensor(six(init),device=D,dtype=torch.float32,requires_grad=True);xy=torch.zeros((B,2),device=D,requires_grad=True);scale=torch.full((B,),1.08,device=D,requires_grad=True);t0=torch.tensor(T[f,:3,3],device=D,dtype=torch.float32)[None].expand(B,-1);o=torch.tensor(orig[n],device=D,dtype=torch.float32)[None].expand(B,-1)
  im=torch.tensor(imgs[n],device=D,dtype=torch.float32)[None].expand(B,-1,-1,-1);m=torch.tensor(masks[n],device=D,dtype=torch.float32)[None].expand(B,-1,-1);p=torch.tensor(people[n],device=D,dtype=torch.float32)[None].expand(B,-1,-1);dt=torch.tensor(dist[n],device=D,dtype=torch.float32)[None].expand(B,-1,-1)
  opt=torch.optim.Adam([{'params':[rr],'lr':.025},{'params':[xy],'lr':.20},{'params':[scale],'lr':.005}])
  for it in range(iters):
   R=rot6(rr);t=translation(xy,t0,K);rgb,a=ren.render(R,t,o,scale=scale.clamp(.95,1.18));loss,co=loss_fn(rgb,a,im,m,p,dt)
   # Small image-plane corrections; no depth or geometric scale update.
   total=loss+.0008*xy.square().sum(-1)+2*(scale-1.08).square();opt.zero_grad();total.sum().backward();opt.step()
  with torch.no_grad():
   R=rot6(rr);t=translation(xy,t0,K);rgb,a=ren.render(R,t,o,scale=scale.clamp(.95,1.18));loss,co=loss_fn(rgb,a,im,m,p,dt)
  scales.append(scale.detach().cpu().numpy());frames.append(R.detach().cpu().numpy());trans.append(t.detach().cpu().numpy());scores.append(loss.detach().cpu().numpy());components.append(co.detach().cpu().numpy())
  if n%10==0:print(n,len(idx),'frame',f,'sec',round(time.time()-start,1),'best',np.round(np.sort(scores[-1])[:3],3),flush=True)
 np.savez(out,idx=idx,scale=np.array(scales),R=np.array(frames),t=np.array(trans),loss=np.array(scores),components=np.array(components));print('done',time.time()-start,flush=True)

import json,time,random
import numpy as np,torch
from scipy.spatial.transform import Rotation,Slerp
import cv2

def choose(q):
 R=q['R'].astype(float);t=q['t'];L=q['loss'];F,C=L.shape
 L=L.copy();L[R[:,:,0,0]>-.35]+=1000  # Mesh +X points into the handle. Visible handle is camera-left throughout this clip.
 L[R[:,:,1,2]<0]+=1000  # The video contains lifts and drinking motions, never an inverted mug.
 dp=L[0].copy();back=np.zeros((F,C),int)
 for f in range(1,F):
  # Image evidence is ambiguous under a grasp. Penalize implausible frame-to-frame flips.
  rel=np.einsum('aij,bkj->abik',R[f-1],R[f]);theta=np.arccos(np.clip((np.trace(rel,axis1=2,axis2=3)-1)/2,-1,1));move=np.linalg.norm(t[f-1,:,None,:]-t[f,None,:,:],axis=-1)
  costs=dp[:,None]+.7*theta**2+5*move**2;back[f]=costs.argmin(0);dp=L[f]+costs.min(0)
 ids=np.zeros(F,int);ids[-1]=dp.argmin()
 for f in range(F-1,0,-1):ids[f-1]=back[f,ids[f]]
 return R[np.arange(F),ids],t[np.arange(F),ids],ids

def refine(inputs,candidates,out,iterations=55):
 start=time.time();torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_num_threads(4)
 q=np.load(candidates);rr,tt,ids=choose(q);cam,base,T,orig,imgs,masks,people=data(inputs);K=np.array(cam['K']);F=len(base);idx=q['idx'];ren=Renderer(inputs+'/object_mesh.glb',K)
 rots=Slerp(idx,Rotation.from_matrix(rr))(np.clip(np.arange(F),idx[0],idx[-1])).as_matrix()
 # Interpolate directions only, and restore each original camera distance.
 tinterp=np.stack([np.interp(np.arange(F),idx,tt[:,j]) for j in range(3)],-1);tinterp=tinterp/np.linalg.norm(tinterp,axis=-1,keepdims=True)*np.linalg.norm(T[:,:3,3],axis=-1,keepdims=True)
 xyinit=(tinterp[:,:2]/tinterp[:,2:]-T[:,:2,3]/T[:,2:3,3])*np.diag(K)[:2]
 resultR=[];resultt=[];metrics=[];baseline=[];scales=[];B=24
 for s in range(0,F,B):
  e=min(F,s+B);N=e-s;init=rots[s:e];rrp=torch.tensor(six(init),device='cuda',dtype=torch.float32,requires_grad=True);xyp=torch.tensor(xyinit[s:e],device='cuda',dtype=torch.float32,requires_grad=True)
  scale=torch.full((N,),1.08,device='cuda',requires_grad=True);refR=torch.tensor(init,device='cuda',dtype=torch.float32);t0=torch.tensor(T[s:e,:3,3],device='cuda',dtype=torch.float32);o=torch.tensor(orig[s:e],device='cuda',dtype=torch.float32)
  im=torch.tensor(imgs[s:e],device='cuda',dtype=torch.float32);m=torch.tensor(masks[s:e],device='cuda',dtype=torch.float32);p=torch.tensor(people[s:e],device='cuda',dtype=torch.float32);dist=np.array([np.minimum(cv2.distanceTransform((~mm).astype('uint8'),cv2.DIST_L2,5),12)**2 for mm in masks[s:e]]);dt=torch.tensor(dist,device='cuda',dtype=torch.float32)
  with torch.no_grad():
   rgb,a=ren.render(T[s:e,:3,:3],T[s:e,:3,3],o);loss,co=loss_fn(rgb,a,im,m,p,dt);baseline.extend(torch.cat([loss[:,None],co],-1).cpu().numpy())
  opt=torch.optim.Adam([{'params':[rrp],'lr':.008},{'params':[xyp],'lr':.08},{'params':[scale],'lr':.003}])
  for it in range(iterations):
   R=rot6(rrp);t=translation(xyp,t0,K);rgb,a=ren.render(R,t,o,scale=scale.clamp(.95,1.18));loss,co=loss_fn(rgb,a,im,m,p,dt)
   # The first fit already explains the image. Limit subsequent drift, particularly under occlusion.
   prior=.04*((R-refR)**2).sum((1,2))+.001*xyp.square().sum(-1)
   # Smooth local corrections; physical movement from the selected trajectory is preserved.
   delta=R@refR.transpose(1,2);smooth=.25*((delta[1:]-delta[:-1])**2).sum()+.015*((xyp[1:]-xyp[:-1])**2).sum()
   total=(loss+prior+2*(scale-1.08).square()).sum()+smooth;opt.zero_grad();total.backward();opt.step()
  with torch.no_grad():
   R=rot6(rrp);t=translation(xyp,t0,K);rgb,a=ren.render(R,t,o,scale=scale.clamp(.95,1.18));loss,co=loss_fn(rgb,a,im,m,p,dt)
  scales.extend(scale.detach().cpu().numpy());resultR.extend(R.detach().cpu().numpy());resultt.extend(t.detach().cpu().numpy());metrics.extend(torch.cat([loss[:,None],co],-1).detach().cpu().numpy())
  if s%(B*5)==0:print('refine',s,F,'sec',round(time.time()-start,1),flush=True)
 Rc=np.array(resultR).astype(float);tc=np.array(resultt).astype(float)
 # Remove subpixel framewise fitting jitter with a short symmetric filter in SE(3).
 from scipy.ndimage import gaussian_filter1d
 smoothR=Rotation.from_matrix(Rc).as_quat();
 for f in range(1,F):
  if np.dot(smoothR[f-1],smoothR[f])<0:smoothR[f]*=-1
 smoothR=gaussian_filter1d(smoothR,.8,axis=0,mode='nearest');Rc=Rotation.from_quat(smoothR).as_matrix()
 uv=tc[:,:2]/tc[:,2:];uv=gaussian_filter1d(uv,.65,axis=0,mode='nearest');tc=np.concatenate([uv,np.ones((F,1))],-1);tc=tc/np.linalg.norm(tc,axis=-1,keepdims=True)*np.linalg.norm(T[:,:3,3],axis=-1,keepdims=True)
 Tout=np.tile(np.eye(4),(F,1,1));Tout[:,:3,:3]=Rc;Tout[:,:3,3]=tc;Tout=np.array(cam['T_world_cam'])@Tout
 assert Tout.shape == base.shape and np.isfinite(Tout).all()
 assert np.max(np.abs(Tout[:,:3,:3]@Tout[:,:3,:3].transpose(0,2,1)-np.eye(3))) < 1e-6
 assert np.max(np.abs(np.linalg.det(Tout[:,:3,:3])-1)) < 1e-6
 camcheck=np.linalg.inv(np.array(cam['T_world_cam']))@Tout
 assert np.max(np.abs(np.linalg.norm(camcheck[:,:3,3],axis=-1)-np.linalg.norm(T[:,:3,3],axis=-1))) < 1e-10
 np.savez_compressed(out,T_world_mesh=Tout.astype(np.float64))
 np.savez(str(Path(out).parent/'fit_diagnostics.npz'),scale=np.array(scales),baseline=np.array(baseline),final=np.array(metrics),selected=ids,key_idx=idx,key_R=rr,key_t=tt)
 print('refine done',time.time()-start,flush=True)

if __name__ == '__main__':
 from pathlib import Path
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--inputs',required=True)
 parser.add_argument('--out',required=True)
 args=parser.parse_args()
 destination=Path(args.out)
 destination.parent.mkdir(parents=True,exist_ok=True)
 candidates=str(destination.parent/'solver_candidates.npz')
 started=time.time()
 torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 fit(args.inputs,candidates,step=4,iters=55)
 refine(args.inputs,candidates,args.out,iterations=55)
 elapsed=time.time()-started
 with open(destination.parent/'runtime.json','w') as stream:
  json.dump({'elapsed_seconds':elapsed,'seed':0,'candidate_stride':4,'candidate_azimuths':12,'candidate_iterations':55,'refinement_iterations':55},stream,indent=2)
 print('Total solver seconds:',round(elapsed,3))
