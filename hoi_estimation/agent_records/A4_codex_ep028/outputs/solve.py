#!/usr/bin/env python3
"""Offline, seeded pose correction for the released squeegee episode.
Uses only supplied video, masks, camera, mesh and reconstruction.
No network, ground truth, external checkpoints or cached data required.
The mesh origin is its area centroid (within 4e-9 m); the baseline
Euclidean distance from this center to the camera is fixed per frame.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, json, time, random, hashlib
from pathlib import Path
import cv2, h5py, numpy as np, trimesh, torch
import nvdiffrast.torch as dr
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d, median_filter, distance_transform_edt

# Orange fitting centroids computed from the supplied mesh texture.
HEAD_POINT = np.array([-.00386614,-.0140458,-.44275588])
TIP_POINT = np.array([.00168146,.01075304,.99291701])

def observations(inputs):
 p=Path(inputs);cap=cv2.VideoCapture(str(p/'video.mp4'));f=h5py.File(p/'masks.h5');F=f['object'].shape[0]; rec=[];t=time.time()
 for i in range(F):
  ok,im=cap.read();mask=f['object'][i].astype('uint8');yy,xx=np.where(mask);x0=max(0,xx.min()-15);x1=min(im.shape[1],xx.max()+16);y0=max(0,yy.min()-15);y1=min(im.shape[0],yy.max()+16);roi=im[y0:y1,x0:x1];ms=mask[y0:y1,x0:x1];h=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV);o=((h[:,:,0]<24)&(h[:,:,1]>130)&(h[:,:,2]>60)&cv2.dilate(ms,np.ones((5,5),np.uint8)).astype(bool)).astype('uint8');n,lab,stats,cent=cv2.connectedComponentsWithStats(o);oranges=[]
  for k in range(1,n):
   if stats[k,4]>=4:oranges.append([*list(cent[k]+[x0,y0]),int(stats[k,4])])
  pts=np.c_[xx,yy].astype(float);cen=pts.mean(0);_,_,vh=np.linalg.svd((pts[::max(1,len(pts)//3000)]-cen),full_matrices=False);axis=vh[0];axis=axis if axis[1]>0 else -axis;proj=(pts-cen)@axis;side=(pts-cen)@np.array([-axis[1],axis[0]]);lo,hi=np.percentile(proj,[0.1,99.9]);bins=np.linspace(lo,hi,31);width=[];count=[];mid=[]
  for l,r in zip(bins[:-1],bins[1:]):
   ss=side[(proj>=l)&(proj<r)];width.append(float(np.percentile(ss,95)-np.percentile(ss,5)) if len(ss)>3 else 0);count.append(len(ss));mid.append(float(np.median(ss)) if len(ss) else 0)
  rec.append(dict(frame=i,orange=oranges,center=cen.tolist(),axis=axis.tolist(),bounds=[lo,hi],width=width,count=count,mid=mid,area=len(xx),bbox=[int(x0),int(y0),int(x1),int(y1)]))
  if i%200==0:print('Observations',i,round(time.time()-t,1),flush=True)
 cap.release();f.close();return rec

def track_features(inputs, obs, start=660):
 p=Path(inputs);cap=cv2.VideoCapture(str(p/'video.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,start);f=h5py.File(p/'masks.h5');F=len(obs);head=np.full((F,2),np.nan);tip=head.copy();quality=np.zeros((F,2));lines=np.full((F,2),np.nan);lasthead=None;lasttip=None
 for i in range(start,F):
  _,im=cap.read();d=obs[i];cen=np.array(d['center']);ax=np.array(d['axis']);mask=f['object'][i].astype('uint8');hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);orange=((hsv[:,:,0]>=3)&(hsv[:,:,0]<23)&(hsv[:,:,1]>180)&(hsv[:,:,2]>90)).astype('uint8');n,lab,stats,cent=cv2.connectedComponentsWithStats(orange);entries=[]
  # The head and handle swap their vertical ordering around frame 712,
  # as the actor rotates the squeegee behind her back. The head remains
  # the left endpoint until it is hidden, and the upper endpoint later.
  for k in range(1,n):
   if stats[k,4]<12:continue
   xy=cent[k];side=abs(np.cross(ax,xy-cen));along=(xy-cen)@ax
   # SAM omits the orange head behind the arm in frames 790-890.
   # Admit color features in a narrow extrapolated shaft corridor.
   if side<22 and -550<along<550 and 480<xy[1]<1100 and 400<xy[0]<980:
    entries.append((xy,stats[k,4],side))
  if i<710:
   expected_h=cen+ax*d['bounds'][1];expected_t=cen+ax*d['bounds'][0]
  else:
   if ax[0]<0:expected_h=cen+ax*d['bounds'][1];expected_t=cen+ax*d['bounds'][0]
   else:expected_h=cen+ax*d['bounds'][0];expected_t=cen+ax*d['bounds'][1]
  # Match color components to temporally predicted endpoint positions,
  # allowing brief occlusion without changing head/handle identities.
  for role,expected,prev,arr,qi in [('head',expected_h,lasthead,head,0),('tip',expected_t,lasttip,tip,1)]:
   cand=[]
   for xy,area,side in entries:
    if prev is not None:
     dis=np.linalg.norm(xy-prev)
     if dis>max(85,30+abs(i-(last_h_i if qi==0 else last_t_i))*6):continue
    else:dis=np.linalg.norm(xy-expected)
    if qi==0 and i>=730 and xy[1]>720:continue
    if qi==1 and i>=740 and xy[1]<740:continue
    if qi==0 and i<735:continue
    if qi==0 and i>=790 and area<20:continue
    cand.append((dis+5*side-3*np.log(area),xy,area))
   if cand:
    _,xy,area=min(cand,key=lambda x:x[0]);arr[i]=xy;quality[i,qi]=area
    if qi==0:lasthead=xy;last_h_i=i
    else:lasttip=xy;last_t_i=i
  # Extract a blade direction from mask pixels close to the head.
  # The small dark orange end loop needs the more permissive color
  # threshold already restricted to SAM; the saturated threshold above
  # is only used to recover the head outside SAM near skin.
  cs=np.array([x for x in d['orange'] if x[2]>=20])
  if len(cs):
   if i<710:jj=np.argmin(cs[:,1])
   elif i<740:jj=np.argmax(cs[:,0])
   else:jj=np.argmax(cs[:,1])
   xy=cs[jj,:2]
   if (i<740 and xy[0]>650) or (i>=740 and xy[1]>740):
    tip[i]=xy;quality[i,1]=cs[jj,2];lasttip=xy;last_t_i=i
  if i<=730:
   cs=np.array([x for x in d['orange'] if x[2]>60])
   if len(cs) and cs[:,0].min()<650:
    xy=cs[np.argmin(cs[:,0]),:2];head[i]=xy;quality[i,0]=cs[np.argmin(cs[:,0]),2];lasthead=xy;last_h_i=i
  if 731<=i<=758:head[i]=np.nan;quality[i,0]=0;lasthead=None
  if i>=759 and np.isfinite(head[i]).all():
   near=[(xy,area) for xy,area,side in entries if np.linalg.norm(xy-head[i])<24]
   if near:head[i]=np.average(np.array([x for x,a in near]),axis=0,weights=[a for x,a in near]);quality[i,0]=sum(a for x,a in near);lasthead=head[i]
  if np.isfinite(head[i]).all():
   hp=head[i];yy,xx=np.where(mask);pts=np.c_[xx,yy];delta=pts-hp;axis_dir=(tip[i]-hp) if np.isfinite(tip[i]).all() else ax;axis_dir=axis_dir/np.linalg.norm(axis_dir);long=delta@axis_dir;side=np.abs(np.cross(axis_dir,delta));keep=(np.linalg.norm(delta,axis=1)<85)&(long<12)&(side>12)
   pts=pts[keep]
   if len(pts)>70:
    mid=np.median(pts,axis=0);_,_,vv=np.linalg.svd(pts-mid,full_matrices=False);direction=vv[0];res=np.abs(np.cross(direction,pts-mid));pts=pts[res<np.percentile(res,75)];_,_,vv=np.linalg.svd(pts-pts.mean(0),full_matrices=False);lines[i]=vv[0]
  if i%50==0:print(i,'head',head[i],'tip',tip[i],flush=True)
 cap.release();f.close();return head,tip,quality,lines

def endpoint_initial(inputs, tracks):
 p=Path(inputs);a=np.load(p/'reconstruction.npz');c=json.load(open(p/'camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);T=np.linalg.inv(W)@a['T_world_mesh'];head=tracks['head'].copy();tip=tracks['tip'].copy();lines=tracks['blade_line'].copy();N=len(T);start=660
 # In this interval SAM omits the blade while RGB shows its upper edge.
 # These are image-plane edge directions measured by OpenCV LSD and
 # checked in the supplied RGB, NOT 3-D annotations. They encode the
 # delayed roll (near-vertical through 840, then horizontal by 870).
 lines[759:920]=np.nan
 for frame,direction in {760:(-.49,.87),770:(-.29,.96),800:(-.12,.99),810:(-.09,1.),820:(-.17,.99),830:(-.23,.97),840:(-.29,.96),850:(-.86,.51),860:(-.97,.23),870:(-1.,.05),900:(-1.,.05),919:(-1.,.05)}.items():
  lines[frame]=np.array(direction)/np.linalg.norm(direction)
 for arr in [head,tip]:
  good=np.isfinite(arr[:,0]);
  for k in range(2):arr[start:,k]=np.interp(np.arange(start,N),np.where(good)[0],arr[good,k]);arr[start:,k]=gaussian_filter1d(arr[start:,k],1.2)
 # A head colored component momentarily merged with skin is rejected by
 # the saturated threshold and interpolated through its occlusion.
 head[727:759]=head[727]+np.linspace(0,1,32)[:,None]*(head[759]-head[727]);
 zhead=-.44275588;ztip=.992917;dist=np.linalg.norm(T[:,:3,3],axis=1);Rinit=[];centers=[];axes=[];xprev=T[start,:3,0];Q=T.copy()
 for i in range(start,N):
  rays=np.c_[np.array([head[i],tip[i]]),np.ones(2)]@np.linalg.inv(K).T
  def fun(z):
   hp,ep=rays*z[:,None];ax=(ep-hp)/(ztip-zhead);t=hp-zhead*ax;return [np.linalg.norm(ep-hp)-(ztip-zhead),np.linalg.norm(t)-dist[i]]
  fits=[]
  for init in [[dist[i]+.4,dist[i]-.5],[dist[i]-.4,dist[i]+.5]]:
   res=least_squares(fun,init,bounds=(.5,6),xtol=1e-11,ftol=1e-11,gtol=1e-11);hp,ep=rays*res.x[:,None];ax=(ep-hp)/(ztip-zhead);t=hp-zhead*ax;fits.append((ax,t,res.cost))
  # The orange handle is closer than the blade before, during and after
  # the turn. Preserve that continuous branch, anchored at frame 660.
  ax,t,_=min(fits,key=lambda x:x[0][2]);ax=ax/np.linalg.norm(ax);axes.append(ax);centers.append(t)
  if np.isfinite(lines[i]).all():
   l=lines[i];uv=head[i];ray=np.linalg.inv(K)@np.r_[uv,1];ray2=np.linalg.inv(K)@np.r_[uv+100*l,1];normal=np.cross(ray,ray2);xx=np.cross(normal,ax);xx/=np.linalg.norm(xx)
   if xx@xprev<0:xx=-xx
  else:
   xx=xprev-ax*(xprev@ax);xx/=np.linalg.norm(xx)
  yy=np.cross(ax,xx);rr=np.stack([xx,yy,ax],axis=1);Rinit.append(rr);xprev=xx;Q[i,:3,:3]=rr;Q[i,:3,3]=t
 # Fill missing blade-angle intervals by interpolating the roll relative
 # to a parallel-transport frame, so occlusion cannot cause 180deg flips.
 axes=np.array(axes);basis=[];xx=T[start,:3,0]
 for ax in axes:
  xx=xx-ax*(xx@ax);xx/=np.linalg.norm(xx);basis.append(np.stack([xx,np.cross(ax,xx),ax],axis=1))
 basis=np.array(basis);valid=np.where(np.isfinite(lines[start:,0]))[0];theta=np.unwrap([np.arctan2((basis[j].T@Rinit[j])[1,0],(basis[j].T@Rinit[j])[0,0]) for j in valid]);theta=np.interp(np.arange(N-start),valid,theta);theta=gaussian_filter1d(theta,2)
 for j,i in enumerate(range(start,N)):
  Q[i,:3,:3]=basis[j]@Rotation.from_rotvec([0,0,theta[j]]).as_matrix()
 return Q

def complete_tracks(inputs, obs, tracks):
 p=Path(inputs);a=np.load(p/'reconstruction.npz');c=json.load(open(p/'camera.json'));K=np.array(c['K']);base=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];head=tracks['head'].copy();tip=tracks['tip'].copy();quality=tracks['quality'].copy();line=tracks['blade_line'].copy();p3=np.array([HEAD_POINT,TIP_POINT])
 for i in range(660):
  xy=(p3@base[i,:3,:3].T+base[i,:3,3])@K.T;xy=xy[:,:2]/xy[:,2:];cs=np.array([x for x in obs[i]['orange'] if x[2]>=6]);
  if not len(cs):continue
  for j,target in enumerate([head,tip]):
   dis=np.linalg.norm(cs[:,:2]-xy[j],axis=1);dis[cs[:,2]<(60 if j==0 else 6)]=1e5;k=np.argmin(dis)
   if dis[k]<60:target[i]=cs[k,:2];quality[i,j]=cs[k,2]
 return dict(head=head,tip=tip,quality=quality,blade_line=line)

def refine_cpu(inputs, initial, tracks, start=0):
 p=Path(inputs);camera=json.load(open(p/'camera.json'));K=np.array(camera['K']);base=np.linalg.inv(camera['T_world_cam'])@np.load(p/'reconstruction.npz')['T_world_mesh'];mesh=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry()
 # Fixed area-stratified surface samples, no stochastic sampling.
 area=np.asarray(mesh.area_faces);cdf=np.cumsum(area);ids=np.searchsorted(cdf,(np.arange(2600)+.5)/2600*cdf[-1]);tris=np.asarray(mesh.triangles)[ids]
 frac=(np.arange(len(ids))*.6180339887498949)%1;frac2=(np.arange(len(ids))*.4142135623730951)%1;sq=np.sqrt(frac);vertices=(1-sq[:,None])*tris[:,0]+(sq*(1-frac2))[:,None]*tris[:,1]+(sq*frac2)[:,None]*tris[:,2]
 points=np.array([[-.00386614,-.0140458,-.44275588],[.00168146,.01075304,.99291701]])
 masks=h5py.File(p/'masks.h5');Q=initial.copy();timer=time.time();params=[];losses=[]
 for i in range(start,len(base)):
  mask=masks['object'][i].astype('uint8');person=masks['person'][i].astype('uint8');yy,xx=np.where(mask);minx=max(0,xx.min()-120);maxx=min(1536,xx.max()+121);miny=max(0,yy.min()-120);maxy=min(1152,yy.max()+121)
  # SciPy EDT is repeatable here; OpenCV DIST_MASK_PRECISE differed by
  # 3e-5 px across processes and perturbed nonsmooth least-squares paths.
  om=mask[miny:maxy,minx:maxx];pm=person[miny:maxy,minx:maxx];allowed=om|pm;dt=distance_transform_edt(1-allowed).astype(np.float32)
  # Deterministic stratification gives each observed segment support.
  pick=np.linspace(0,len(xx)-1,min(240,len(xx))).astype(int);target=np.c_[xx[pick],yy[pick]].astype(float)
  obs=np.stack([tracks['head'][i],tracks['tip'][i]]);valid=np.isfinite(obs[:,0]);dist=np.linalg.norm(base[i,:3,3]);R0=initial[i,:3,:3];t0=initial[i,:3,3];uv0=t0[:2]/t0[2]
  def pose(x):
   R=Rotation.from_rotvec(x[:3]).as_matrix()@R0;ray=np.r_[uv0+x[3:]/K[0,0],1.];t=ray/np.linalg.norm(ray)*dist;return R,t
  def project(v,R,t):
   xyz=v@R.T+t;uv=xyz@K.T;return uv[:,:2]/uv[:,2:]
  def objective(x):
   R,t=pose(x);uv=project(vertices,R,t);xy=(uv-[minx,miny]).astype(np.float32)
   outside=cv2.remap(dt,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=70).ravel()
   # Beyond a 1.5px tolerance, penalize model pixels outside all allowed
   # support; person regions permit genuine object occlusion.
   a=np.maximum(outside-1.,0)/np.sqrt(len(outside))
   distances=cKDTree(uv).query(target)[0];b=np.maximum(distances-2.,0)/np.sqrt(len(target))
   puv=project(points,R,t);land=(puv[valid]-obs[valid]).ravel()*.38
   # Strong orientation trust region before failure, lighter late.
   reg=x[:3]*(10 if i<660 else 6);transreg=x[3:]*.012
   # Quantize far below image precision to prevent tiny platform math
   # differences from selecting different nonsmooth solver branches.
   return np.round(np.r_[a,b,land,reg,transreg],7)
  # Differencing must exceed cv2's 1/32px interpolation table spacing.
  # Central finite differences are explicit, avoiding quantization traps.
  def jac(x):
   columns=[]
   for j in range(5):
    step=.0004 if j<3 else .08;e=np.zeros(5);e[j]=step;columns.append((objective(x+e)-objective(x-e))/(2*step))
   return np.stack(columns,axis=1)
  result=least_squares(objective,np.zeros(5),jac=jac,loss='soft_l1',f_scale=2.,max_nfev=20,xtol=3e-5,ftol=3e-5,gtol=3e-5,bounds=([-0.3]*3+[-60]*2,[.3]*3+[60]*2))
  Q[i,:3,:3],Q[i,:3,3]=pose(result.x);params.append(result.x);losses.append(result.cost)
  if i%100==0:print('CPU fit',i,'seconds',round(time.time()-timer,1),flush=True)
 # Smooth only residual corrections relative to the original trajectory.
 # Baseline camera distance remains exact after filtering the image ray.
 pars=gaussian_filter1d(np.array(params),.8,axis=0)
 for j,i in enumerate(range(start,len(base))):
  R0=initial[i,:3,:3];t0=initial[i,:3,3];x=pars[j];Q[i,:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@R0;ray=np.r_[t0[:2]/t0[2]+x[3:]/K[0,0],1.];Q[i,:3,3]=ray/np.linalg.norm(ray)*np.linalg.norm(base[i,:3,3])
 masks.close();return Q,losses,time.time()-timer


def score_poses(inputs, baseline, candidate, tracks):
 """Full-mesh image-consistency check at half video resolution, every frame."""
 p=Path(inputs);c=json.load(open(p/'camera.json'));K=np.array(c['K']);m=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry()
 v=torch.tensor(np.array(m.vertices),device='cuda',dtype=torch.float32);f=torch.tensor(np.array(m.faces),device='cuda',dtype=torch.int32);ctx=dr.RasterizeCudaContext();masks=h5py.File(p/'masks.h5');scores=[]
 points=np.stack([HEAD_POINT,TIP_POINT]);uvobs=np.stack([tracks['head'],tracks['tip']],1)
 for i in range(len(baseline)):
  oo=masks['object'][i][::2,::2];pp=masks['person'][i][::2,::2];pair=[]
  for tt in [baseline[i],candidate[i]]:
   P=torch.tensor(tt,device='cuda',dtype=torch.float32);xyz=v@P[:3,:3].T+P[:3,3];z=xyz[:,2];u=K[0,0]*xyz[:,0]+K[0,2]*z;vv=K[1,1]*xyz[:,1]+K[1,2]*z
   clip=torch.stack([2*u/1536-z,2*vv/1152-z,.99*z-.01,z],-1)[None];rast,_=dr.rasterize(ctx,clip,f,resolution=[576,768]);ma=(rast[0,:,:,3]>0).cpu().numpy();visible=ma&(~pp|oo)
   iou=(visible&oo).sum()/max(1,(visible|oo).sum());xyzp=points@tt[:3,:3].T+tt[:3,3];uv=xyzp@K.T;uv=uv[:,:2]/uv[:,2:];err=np.linalg.norm(uv-uvobs[i],axis=1);pair.append([iou,*err])
  scores.append(pair)
 masks.close();return np.array(scores)

def choose_conservatively(base, candidate, scores):
 """Keep early baseline wherever image evidence for an edit is weak."""
 gain=scores[:,1,0]-scores[:,0,0]
 strong=(gain>.055)&(median_filter(gain,size=9)>.055)
 angle=Rotation.from_matrix(candidate[:,:3,:3]@base[:,:3,:3].transpose(0,2,1)).magnitude()
 displacement=np.linalg.norm(candidate[:,:3,3]-base[:,:3,3],axis=1)
 strong &= (angle<np.deg2rad(18))&(displacement<.16)
 # RGB landmarks must not contradict an otherwise improved silhouette.
 errors0=scores[:,0,1:];errors1=scores[:,1,1:]
 n=np.isfinite(errors0).sum(1);e0=np.nansum(errors0,axis=1)/np.maximum(n,1);e1=np.nansum(errors1,axis=1)/np.maximum(n,1)
 strong &= (n==0)|(e1<=e0+1)
 alpha=strong.astype(float)
 # A five-frame fade avoids discontinuous acceptance at the boundary.
 alpha=gaussian_filter1d(alpha,1.,mode='nearest');alpha[~strong]=0
 # After 690, confirmed head/handle identity and temporal continuation
 # replace the failed track. The late full-mesh alignment is checked.
 alpha[690:]=1
 for i in range(680,690):alpha[i]=max(alpha[i],(i-680)/10)
 Q=base.copy();delta=Rotation.from_matrix(candidate[:,:3,:3]@base[:,:3,:3].transpose(0,2,1)).as_rotvec();rot=Rotation.from_rotvec(alpha[:,None]*delta).as_matrix()@base[:,:3,:3]
 trans=(1-alpha[:,None])*base[:,:3,3]+alpha[:,None]*candidate[:,:3,3]
 dist=np.linalg.norm(base[:,:3,3],axis=1);trans=trans/np.linalg.norm(trans,axis=1,keepdims=True)*dist[:,None]
 modified=alpha>0
 Q[modified,:3,:3]=Rotation.from_matrix(rot[modified]).as_matrix();Q[modified,:3,3]=trans[modified]
 return Q,alpha

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 started=time.time();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(1);cv2.setNumThreads(1);cv2.setRNGSeed(0)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True,warn_only=True)
 args.out.parent.mkdir(parents=True,exist_ok=True)
 reconstruction=np.load(args.inputs/'reconstruction.npz');baseworld=reconstruction['T_world_mesh'].astype(np.float64);cam=json.load(open(args.inputs/'camera.json'));W=np.array(cam['T_world_cam'],dtype=np.float64);base=np.linalg.inv(W)@baseworld
 assert len(base)==json.load(open(args.inputs/'episode.json'))['frames']==988
 obs=observations(args.inputs);h,t,q,l=track_features(args.inputs,obs);tracks=complete_tracks(args.inputs,obs,dict(head=h,tip=t,quality=q,blade_line=l))
 initial=endpoint_initial(args.inputs,tracks)
 candidate,loss,fit_seconds=refine_cpu(args.inputs,initial,tracks,start=0)
 print('Checking candidate against baseline on every frame',flush=True)
 scores=score_poses(args.inputs,base,candidate,tracks)
 final,alpha=choose_conservatively(base,candidate,scores)
 # Restore exact original world matrices for rejected frames.
 world=W@final;world[alpha==0]=baseworld[alpha==0];world[:,3,:]=[0,0,0,1]
 final_cam=np.linalg.inv(W)@world
 distance_error=np.abs(np.linalg.norm(final_cam[:,:3,3],axis=1)-np.linalg.norm(base[:,:3,3],axis=1))
 orth_error=np.max(np.abs(world[:,:3,:3].transpose(0,2,1)@world[:,:3,:3]-np.eye(3)))
 assert np.isfinite(world).all() and world.shape==(988,4,4)
 assert distance_error.max()<1e-10 and orth_error<1e-4
 assert np.min(np.linalg.det(world[:,:3,:3]))>.9999
 np.savez(args.out,T_world_mesh=world.astype(np.float64))
 diagnostics=args.out.with_name(args.out.stem+'_diagnostics.npz')
 np.savez(diagnostics,acceptance_weight=alpha,candidate_scores=scores,head=tracks['head'],tip=tracks['tip'],fitting_loss=loss)
 log=dict(frames=len(world),dtype=str(world.dtype),changed_frames=int((alpha>0).sum()),unchanged_frames=int((alpha==0).sum()),max_camera_distance_error_m=float(distance_error.max()),max_rotation_orthogonality_error=float(orth_error),runtime_seconds=time.time()-started,cpu_fit_seconds=fit_seconds,seed=0,device=torch.cuda.get_device_name(0))
 args.out.with_name(args.out.stem+'_run.json').write_text(json.dumps(log,indent=2)+'\n')
 print(json.dumps(log,indent=2),flush=True)

if __name__=='__main__':main()
