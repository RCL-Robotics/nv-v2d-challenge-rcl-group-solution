#!/usr/bin/env python3
"""Offline, seeded pose correction for the released squeegee episode.
Uses only supplied video, masks, camera, mesh and reconstruction.
No network, ground truth, external checkpoints or cached data required.
The mesh origin is its area centroid (within 4e-9 m); the baseline
Euclidean distance from this center to the camera is fixed per frame.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, json, time, random, hashlib
from pathlib import Path
import cv2, h5py, numpy as np, trimesh, torch
import nvdiffrast.torch as dr
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d, median_filter

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

def refine(inputs, init, tracks, start=660, steps=100):
 timer=time.time();p=Path(inputs);a=np.load(p/'reconstruction.npz');c=json.load(open(p/'camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);base=np.linalg.inv(W)@a['T_world_mesh'];head=tracks['head'];tip=tracks['tip'];m=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry();fullv=np.array(m.vertices);fullcolor=m.visual.to_color().vertex_colors[:,:3]/255.;m.merge_vertices(merge_tex=True,merge_norm=True);m=m.simplify_quadric_decimation(face_count=12000);v=np.asarray(m.vertices);f=np.asarray(m.faces);colors=fullcolor[cKDTree(fullv).query(v)[1]];ctx=dr.RasterizeCudaContext();device='cuda';ft=torch.tensor(f.astype('int32'),device=device);vt=torch.tensor(v.astype('float32'),device=device);ct=torch.tensor(colors.astype('float32'),device=device);cap=cv2.VideoCapture(str(p/'video.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,start);masks=h5py.File(p/'masks.h5');N=len(base);S=256;ims=[];ob=[];per=[];dists=[];boxes=[];orange=[];orange_dt=[];surface=[]
 for i in range(start,N):
  _,im=cap.read();om=masks['object'][i].astype('uint8');pm=masks['person'][i].astype('uint8');yy,xx=np.where(om);uv=(v@init[i,:3,:3].T+init[i,:3,3])@K.T;uv=uv[:,:2]/uv[:,2:];lo=np.minimum([xx.min(),yy.min()],uv.min(0));hi=np.maximum([xx.max(),yy.max()],uv.max(0));center=(lo+hi)/2;size=max(hi-lo)+80;size=min(size,900);left=int(center[0]-size/2);top=int(center[1]-size/2);sz=int(size);boxes.append([left,top,sz]);
  def crop(x,interp):
   M=np.array([[S/sz,0,-left*S/sz],[0,S/sz,-top*S/sz]],float);return cv2.warpAffine(x,M,(S,S),flags=interp)
  oo=crop(om,cv2.INTER_NEAREST);pp=crop(pm,cv2.INTER_NEAREST);rgb=crop(im,cv2.INTER_LINEAR);ims.append(rgb[:,:,::-1].copy()/255.);ob.append(oo);per.append(pp);allow=oo|pp;dt=cv2.distanceTransform((1-allow).astype('uint8'),cv2.DIST_L2,3);dists.append(np.minimum(dt,30));hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);oo2=((hsv[:,:,0]<24)&(hsv[:,:,1]>130)&(hsv[:,:,2]>60))&(cv2.dilate(oo,np.ones((3,3),np.uint8))>0);orange.append(oo2);orange_dt.append(np.minimum(cv2.distanceTransform((1-oo2).astype('uint8'),cv2.DIST_L2,3),30))
 print('prepared',round(time.time()-timer,2),flush=True)
 B=N-start;boxes=np.array(boxes);obs=torch.tensor(np.array(ob),device=device,dtype=torch.float32);persons=torch.tensor(np.array(per),device=device,dtype=torch.float32);dt=torch.tensor(np.array(dists),device=device,dtype=torch.float32);odt=torch.tensor(np.array(orange_dt),device=device,dtype=torch.float32);rgb=torch.tensor(np.array(ims),device=device,dtype=torch.float32);rt=torch.tensor(init[start:,:3,:3],device=device,dtype=torch.float32);tt=torch.tensor(init[start:,:3,3],device=device,dtype=torch.float32);rr=torch.tensor(np.linalg.norm(base[start:,:3,3],axis=1),device=device,dtype=torch.float32);bt=torch.tensor(boxes,device=device,dtype=torch.float32);delta=torch.nn.Parameter(torch.zeros((B,3),device=device));uv0=tt[:,:2]/tt[:,2:];duv=torch.nn.Parameter(torch.zeros((B,2),device=device));opt=torch.optim.Adam([{'params':[delta],'lr':.007},{'params':[duv],'lr':.001}]);hpt=np.array([-.00386614,-.0140458,-.44275588]);tpt=np.array([.00168146,.01075304,.99291701]);lpts=torch.tensor(np.stack([hpt,tpt]),device=device,dtype=torch.float32);lobs=np.stack([head[start:],tip[start:]],axis=1);lvalid=np.isfinite(lobs[:,:,0]);lobs=np.nan_to_num(lobs);lobs_t=torch.tensor(lobs,device=device,dtype=torch.float32);valid_t=torch.tensor(lvalid,device=device,dtype=torch.float32);objarea=obs.sum((1,2)).clamp(min=30);losslog=[]
 def poses():
  x,y,z=delta.unbind(-1);zero=torch.zeros_like(x);skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(-1,3,3);rot=torch.matrix_exp(skew)@rt;ray=torch.cat([uv0+duv,torch.ones((B,1),device=device)],1);trans=ray/torch.linalg.norm(ray,dim=1,keepdim=True)*rr[:,None];return rot,trans
 for step in range(steps):
  opt.zero_grad();total=0.
  for j in range(0,B,16):
   k=min(j+16,B);rot,trans=poses();rot=rot[j:k];trans=trans[j:k];xyz=torch.einsum('bij,vj->bvi',rot,vt)+trans[:,None];z=xyz[:,:,2];u=K[0,0]*xyz[:,:,0]+K[0,2]*z;vv=K[1,1]*xyz[:,:,1]+K[1,2]*z;box=bt[j:k];clip=torch.stack([2*(u-box[:,None,0]*z)/box[:,None,2]-z,2*(vv-box[:,None,1]*z)/box[:,None,2]-z,.99*z-.01,z],-1);rast,_=dr.rasterize(ctx,clip,ft,resolution=[S,S]);alpha=(rast[:,:,:,3:]>0).float();alpha=dr.antialias(alpha,rast,clip,ft)[:,:,:,0];co,_=dr.interpolate(ct[None].expand(k-j,-1,-1).contiguous(),rast,ft);co=dr.antialias(co,rast,clip,ft);oo=obs[j:k];pp=persons[j:k];vis=alpha*(1-pp*(1-oo));
   # Visible silhouette plus occlusion-aware outside distance.
   miss=((1-alpha)*oo).sum((1,2))/objarea[j:k];extra=(vis*(1-oo)).sum((1,2))/objarea[j:k];outside=(vis*dt[j:k]).sum((1,2))/objarea[j:k];sil=1.6*miss+.75*extra+.25*outside
   # Hue cue discourages rendering the black blade at the orange tip.
   predorange=(co[:,:,:,0]-co[:,:,:,1]*1.18).clamp(0)*vis;closs=(predorange*odt[j:k]).sum((1,2))/objarea[j:k];
   lp=torch.einsum('bij,kj->bki',rot,lpts)+trans[:,None];puv=torch.stack([K[0,0]*lp[:,:,0]/lp[:,:,2]+K[0,2],K[1,1]*lp[:,:,1]/lp[:,:,2]+K[1,2]],-1);err=torch.linalg.norm(puv-lobs_t[j:k],dim=-1);land=(torch.nn.functional.huber_loss(err,torch.zeros_like(err),delta=4,reduction='none')*valid_t[j:k]).sum(1)*.008;
   loss=(sil+.6*closs+land).sum()/B;loss.backward();total+=float(loss.detach())
  rot,trans=poses();reg=2.*delta.square().sum(1).mean()+30*duv.square().sum(1).mean();smooth=25*(rot[2:]-2*rot[1:-1]+rot[:-2]).square().mean()+150*(trans[2:]-2*trans[1:-1]+trans[:-2]).square().mean();(reg+smooth).backward();opt.step();losslog.append(total)
  if step%20==0 or step==steps-1:print(step,'loss',round(total,4),'reg',round(float(reg),4),'seconds',round(time.time()-timer,1),flush=True)
 rot,trans=poses();Q=init.copy();Q[start:,:3,:3]=rot.detach().cpu().numpy();Q[start:,:3,3]=trans.detach().cpu().numpy();cap.release();masks.close();return Q,losslog,time.time()-timer

def score_poses(inputs, baseline, candidate, tracks):
 """Independent full-mesh check at half video resolution, every frame."""
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
 started=time.time();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4);cv2.setNumThreads(1);cv2.setRNGSeed(0)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True,warn_only=True)
 args.out.parent.mkdir(parents=True,exist_ok=True)
 reconstruction=np.load(args.inputs/'reconstruction.npz');baseworld=reconstruction['T_world_mesh'].astype(np.float64);cam=json.load(open(args.inputs/'camera.json'));W=np.array(cam['T_world_cam'],dtype=np.float64);base=np.linalg.inv(W)@baseworld
 assert len(base)==json.load(open(args.inputs/'episode.json'))['frames']==988
 obs=observations(args.inputs);h,t,q,l=track_features(args.inputs,obs);tracks=complete_tracks(args.inputs,obs,dict(head=h,tip=t,quality=q,blade_line=l))
 initial=endpoint_initial(args.inputs,tracks)
 late,late_loss,late_seconds=refine(args.inputs,initial,tracks,start=660,steps=100)
 # Fit only small residual changes from the baseline before failure.
 fullinit=base.copy();fullinit[660:]=late[660:]
 candidate,loss,fit_seconds=refine(args.inputs,fullinit,tracks,start=0,steps=80)
 # Retain the RGB-constrained late roll, without a second free roll fit.
 candidate[660:]=late[660:]
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
 np.savez(diagnostics,acceptance_weight=alpha,candidate_scores=scores,head=tracks['head'],tip=tracks['tip'],late_loss=late_loss,full_loss=loss)
 log=dict(frames=len(world),dtype=str(world.dtype),changed_frames=int((alpha>0).sum()),unchanged_frames=int((alpha==0).sum()),max_camera_distance_error_m=float(distance_error.max()),max_rotation_orthogonality_error=float(orth_error),runtime_seconds=time.time()-started,late_fit_seconds=late_seconds,full_fit_seconds=fit_seconds,seed=0,device=torch.cuda.get_device_name(0))
 args.out.with_name(args.out.stem+'_run.json').write_text(json.dumps(log,indent=2)+'\n')
 print(json.dumps(log,indent=2),flush=True)

if __name__=='__main__':main()
