import json,time,random
import numpy as np,torch
from scipy.spatial.transform import Rotation,Slerp
from fit_poses import data,rot6,six,translation,loss_fn
from render_utils import Renderer
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
 resultR=[];resultt=[];metrics=[];baseline=[];B=24
 for s in range(0,F,B):
  e=min(F,s+B);N=e-s;init=rots[s:e];rrp=torch.tensor(six(init),device='cuda',dtype=torch.float32,requires_grad=True);xyp=torch.tensor(xyinit[s:e],device='cuda',dtype=torch.float32,requires_grad=True)
  refR=torch.tensor(init,device='cuda',dtype=torch.float32);t0=torch.tensor(T[s:e,:3,3],device='cuda',dtype=torch.float32);o=torch.tensor(orig[s:e],device='cuda',dtype=torch.float32)
  im=torch.tensor(imgs[s:e],device='cuda',dtype=torch.float32);m=torch.tensor(masks[s:e],device='cuda',dtype=torch.float32);p=torch.tensor(people[s:e],device='cuda',dtype=torch.float32);dist=np.array([np.minimum(cv2.distanceTransform((~mm).astype('uint8'),cv2.DIST_L2,5),12)**2 for mm in masks[s:e]]);dt=torch.tensor(dist,device='cuda',dtype=torch.float32)
  with torch.no_grad():
   rgb,a=ren.render(T[s:e,:3,:3],T[s:e,:3,3],o);loss,co=loss_fn(rgb,a,im,m,p,dt);baseline.extend(torch.cat([loss[:,None],co],-1).cpu().numpy())
  opt=torch.optim.Adam([{'params':[rrp],'lr':.008},{'params':[xyp],'lr':.08}])
  for it in range(iterations):
   R=rot6(rrp);t=translation(xyp,t0,K);rgb,a=ren.render(R,t,o);loss,co=loss_fn(rgb,a,im,m,p,dt)
   # The first fit already explains the image. Limit subsequent drift, particularly under occlusion.
   prior=.04*((R-refR)**2).sum((1,2))+.001*xyp.square().sum(-1)
   # Smooth local corrections; physical movement from the selected trajectory is preserved.
   delta=R@refR.transpose(1,2);smooth=.25*((delta[1:]-delta[:-1])**2).sum()+.015*((xyp[1:]-xyp[:-1])**2).sum()
   total=(loss+prior).sum()+smooth;opt.zero_grad();total.backward();opt.step()
  with torch.no_grad():
   R=rot6(rrp);t=translation(xyp,t0,K);rgb,a=ren.render(R,t,o);loss,co=loss_fn(rgb,a,im,m,p,dt)
  resultR.extend(R.detach().cpu().numpy());resultt.extend(t.detach().cpu().numpy());metrics.extend(torch.cat([loss[:,None],co],-1).detach().cpu().numpy())
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
 np.savez_compressed(out,T_world_mesh=Tout.astype(np.float64))
 np.savez('/work/outputs/fit_diagnostics.npz',baseline=np.array(baseline),final=np.array(metrics),selected=ids,key_idx=idx,key_R=rr,key_t=tt)
 print('refine done',time.time()-start,flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--candidates',default='/work/outputs/candidates.npz');p.add_argument('--out',default='/work/outputs/object_poses_world.npz');a=p.parse_args();refine('/work/inputs',a.candidates,a.out)
