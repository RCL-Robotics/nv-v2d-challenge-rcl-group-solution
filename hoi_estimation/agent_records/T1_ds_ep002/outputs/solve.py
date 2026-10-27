import argparse,json,os,sys,time
import numpy as np,cv2,h5py,trimesh,torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import differential_evolution,minimize
from scipy.ndimage import gaussian_filter1d

np.random.seed(0); torch.manual_seed(0)
RES=160

class Renderer:
 def __init__(self,mesh_path,device='cuda'):
  self.device=device; self.ctx=dr.RasterizeCudaContext()
  m=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
  self.mesh=m; self.V=np.c_[m.vertices,np.ones(len(m.vertices),np.float32)].astype(np.float32)
  self.F=m.faces.astype(np.int32);self.UV=m.visual.uv.astype(np.float32)
  self.tex=np.array(m.visual.material.baseColorTexture).astype(np.float32)/255.
  self.vh=torch.tensor(self.V,device=device);self.ft=torch.tensor(self.F,dtype=torch.int32,device=device);self.uvt=torch.tensor(self.UV,device=device);self.tt=torch.tensor(self.tex,device=device)
  # sample of vertices for floor checks
  self.hull=m.convex_hull.vertices.astype(np.float32)
 def render_batch(self,poses,Ks,sizes):
  B=len(poses);h,w=int(sizes[0,1]),int(sizes[0,0]);fx,fy=Ks[0,0,0],Ks[0,1,1];cx,cy=Ks[0,0,2],Ks[0,1,2]
  P=np.array([[2*fx/w,0,2*cx/w-1,0],[0,2*fy/h,2*cy/h-1,0],[0,0,.5,0],[0,0,1,0]],np.float32)
  mats=torch.tensor(np.einsum('ij,bjk->bik',P,poses).astype(np.float32),device=self.device)
  clip=torch.einsum('bnj,bkj->bnk',self.vh[None].expand(B,-1,-1),mats)
  rast,rdb=dr.rasterize(self.ctx,clip,self.ft,resolution=[h,w])
  uv,_=dr.interpolate(self.uvt[None].expand(B,-1,-1).contiguous(),rast,self.ft)
  pix=uv.reshape(B,h,w,2)
  col=dr.texture(self.tt[None].expand(B,-1,-1,-1).contiguous(),pix,filter_mode='linear')
  return rast[...,3]>0,col,rdb


def crop_K(K,x0,y0,sz,lim=RES):
 s=lim/sz;Kr=K.copy();Kr[0,0]*=s;Kr[1,1]*=s;Kr[0,2]=(K[0,2]-x0)*s;Kr[1,2]=(K[1,2]-y0)*s;return Kr

def projected_bbox(ren,pose,K):
 pc=(pose@ren.V.T).T[:,:3];uv=(K@pc.T).T;uv=uv[:,:2]/uv[:,2:3];v=(pc[:,2]>0)&np.isfinite(uv).all(1);return uv[v].min(0),uv[v].max(0)

def make_crop(obj,pose,ren,K,W,H,pad=.8,lim=RES):
 ys,xs=np.where(obj);mn,mx=projected_bbox(ren,pose,K)
 x0,x1,y0,y1=mn[0],mx[0],mn[1],mx[1]
 if len(xs):x0=min(x0,xs.min());x1=max(x1,xs.max());y0=min(y0,ys.min());y1=max(y1,ys.max())
 sz=int(np.ceil(max((x1-x0)*(1+2*pad),(y1-y0)*(1+2*pad),64)));cx=int(round((x0+x1)/2));cy=int(round((y0+y1)/2));sx=max(0,min(W-sz,cx-sz//2));sy=max(0,min(H-sz,cy-sz//2));sz=min(sz,W-sx,H-sy);return sx,sy,sz,crop_K(K,sx,sy,sz,lim)

def resized_masks(obj,per,x0,y0,sz,lim=RES):
 o=cv2.resize(obj[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0
 p=cv2.resize(per[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0
 return o,p

def mask_iou(pred,obj,per):
 eff=pred&~per;inter=(eff&obj).sum();uni=(eff|obj).sum();return float(inter/uni) if uni else 0.0

def quat_slerp(q0,q1,t):
 q0=np.asarray(q0,float);q1=np.asarray(q1,float)
 if q0@q1<0:q1=-q1
 if t<=0:return q0/np.linalg.norm(q0)
 if t>=1:return q1/np.linalg.norm(q1)
 th=np.arccos(np.clip(q0@q1,-1,1))
 if th<1e-7:return q0/np.linalg.norm(q0)
 return (np.sin((1-t)*th)*q0+np.sin(t*th)*q1)/np.sin(th)

def matrix_slerp(A,B,t):return Rot.from_quat(quat_slerp(Rot.from_matrix(A).as_quat(),Rot.from_matrix(B).as_quat(),t)).as_matrix()

def robust_orientations(T,lo,hi,start_R=None,end_R=None):
 q=Rot.from_matrix(T[:,:3,:3]).as_quat().copy();seg=q[lo:hi+1].copy()
 # two robust rolling medoid passes
 for _ in range(2):
  out=seg.copy()
  for i in range(len(seg)):
   ids=np.arange(max(0,i-4),min(len(seg),i+5));qs=seg[ids].copy();qs*=np.sign(qs@seg[i])[:,None];w=np.exp(-.5*((ids-i)/2.7)**2)
   D=1-np.abs(qs@qs.T);score=(D*w[:,None]).sum(0);k=int(np.argmin(score));out[i]=quat_slerp(seg[i],qs[k],.72)
  seg=out
 # hemisphere-consistent light smoothing
 for i in range(1,len(seg)):
  if seg[i]@seg[i-1]<0:seg[i]*=-1
 sig=1.0;sm=np.stack([gaussian_filter1d(seg[:,j],sig,mode='nearest') for j in range(4)],1);nrm=np.linalg.norm(sm,axis=1);bad=nrm<1e-8;sm[bad]=seg[bad];nrm[bad]=np.linalg.norm(sm[bad],axis=1);sm/=nrm[:,None]
 if start_R is not None: sm[0]=Rot.from_matrix(start_R).as_quat()
 if end_R is not None: sm[-1]=Rot.from_matrix(end_R).as_quat()
 return Rot.from_matrix(Rot.from_quat(sm).as_matrix()).as_matrix()

def hand_frames(keypoints,idx):
 out=np.zeros((len(keypoints),3,3),np.float32);prev=None
 for t,p in enumerate(keypoints[:,idx]):
  c=p.mean(0);X=p-c;w,V=np.linalg.eigh(X.T@X);V=V[:,::-1];V[:,2]=np.cross(V[:,0],V[:,1]);V[:,1]=np.cross(V[:,2],V[:,0])
  if prev is not None:
   if V[:,0]@prev[:,0]<0:V[:,0]*=-1;V[:,2]*=-1
   if V[:,1]@prev[:,1]<0:V[:,1]*=-1;V[:,2]*=-1
   V[:,2]=np.cross(V[:,0],V[:,1])
  out[t]=V;prev=V
 return out

def align_xy(ren,Tcw,K,obj,per,Rin,tin,iters=4):
 pose0=np.eye(4,dtype=np.float32);pose0[:3,:3]=Rin;pose0[:3,3]=tin
 x0,y0,sz,Kr=make_crop(obj,Tcw@pose0,ren,K,1536,1152,pad=.5)
 objr,perr=resized_masks(obj,per,x0,y0,sz);ys,xs=np.where(objr);tc=np.array([xs.mean(),ys.mean()])
 t=np.array(tin,float)
 for it in range(iters):
  P=np.eye(4,dtype=np.float32);P[:3,:3]=Rin;P[:3,3]=t;m,_,_=ren.render_batch([(Tcw@P).astype(np.float32)],Kr[None],np.array([[RES,RES]],np.float32));m=m[0].cpu().numpy();eff=m&~perr;py,px=np.where(eff)
  if len(px)<10:break
  pc=np.array([px.mean(),py.mean()]);du=(tc-pc)/ (RES/sz);qc=Tcw[:3,:3]@t+Tcw[:3,3];Z=qc[2];dcam=np.array([du[0]*Z/K[0,0],du[1]*Z/K[1,1],0]);t+=Tcw[:3,:3].T@dcam
 return t

def static_refine_xy(ren,Tcw,K,T,pairs,R,t0,bounds=.07):
 # pairs list (frame,obj,per); evaluate fixed crop around representative middle frame
 mid=len(pairs)//2;i,obj,per=pairs[mid];P=np.eye(4,dtype=np.float32);P[:3,:3]=R;P[:3,3]=t0;x0,y0,sz,Kr=make_crop(obj,Tcw@P,ren,K,1536,1152,pad=.8)
 vals=[]
 for j,o,p in pairs:
  orr,prr=resized_masks(o,p,x0,y0,sz);vals.append((j,orr,prr))
 sizes=np.array([[RES,RES]],np.float32)
 def cost(d):
  t=t0.copy();t[:2]+=d;P=np.eye(4,dtype=np.float32);P[:3,:3]=R;P[:3,3]=t;poses=np.repeat((Tcw@P)[None],len(vals),axis=0)
  mm,_,_=ren.render_batch(poses.astype(np.float32),Kr[None],sizes);mm=mm.cpu().numpy();return 1-np.mean([mask_iou(m,o,p) for m,(_,o,p) in zip(mm,vals)])
 res=differential_evolution(cost,[(-bounds,bounds),(-bounds,bounds)],seed=0,popsize=5,maxiter=9,tol=.005,polish=False,disp=False);r2=minimize(cost,res.x,method='Nelder-Mead',options={'maxiter':80});t=t0.copy();t[:2]+=r2.x;return t,1-r2.fun

def anchor_ramp(Rseq,start_R=None,end_R=None,n=12):
 if len(Rseq)<2:return Rseq
 def ss(x):return x*x*(3-2*x)
 if start_R is not None:
  Rseq[0]=start_R
  for k in range(1,min(n,len(Rseq))):
   Rseq[k]=matrix_slerp(start_R,Rseq[k],ss(k/(n-1)))
 if end_R is not None:
  Rseq[-1]=end_R
  for k in range(1,min(n,len(Rseq))):
   Rseq[-1-k]=matrix_slerp(end_R,Rseq[-1-k],ss(k/(n-1)))
 return Rseq

def vec_anchor_ramp(P,start=None,end=None,n=10):
 P=np.array(P,copy=True)
 if len(P)<2:return P
 def ss(x):return x*x*(3-2*x)
 if start is not None:
  P[0]=start
  for k in range(1,min(n,len(P))):
   s=ss(k/(n-1));P[k]=(1-s)*np.asarray(start)+s*P[k]
 if end is not None:
  P[-1]=end
  for k in range(1,min(n,len(P))):
   s=ss(k/(n-1));P[-1-k]=(1-s)*np.asarray(end)+s*P[-1-k]
 return P

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',default='/work/inputs');ap.add_argument('--out',default='/work/outputs/object_poses_world.npz');args=ap.parse_args();tstart=time.time();root=args.inputs;r=np.load(root+'/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);K=np.array(json.load(open(root+'/camera.json'))['K'],np.float32);Twc=np.array(json.load(open(root+'/camera.json'))['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32);pl=np.array(json.load(open(root+'/ground_plane.json'))['plane']);n=pl[:3];d=pl[3];H,W=1152,1536
 ren=Renderer(root+'/object_mesh.glb');R0=T[0,:3,:3].copy()
 # Episode-level phase decisions, verified from full-resolution video, object/person masks,
 # and the contact signal. A/C/E are periods in which the object is at rest.
 A_END=94; B_START=95; B_END=258; C_START=259; C_END=414; D_START=415; D_END=845; E_START=846
 # load selected masks without holding all frames
 objm={};perm={}
 frames_needed=set(range(0,980,2))|set(range(95,259))|set([x for x in range(259,419,5)])|set(range(415,868))|set([745,846,979])
 with h5py.File(root+'/masks.h5','r') as f:
  for i in sorted(frames_needed):
   objm[i]=np.array(f['object'][i]);perm[i]=np.array(f['person'][i])
 # ---------- static A ----------
 tA=np.median(T[0:95,:3,3],axis=0);pairsA=[(i,objm[i],perm[i]) for i in list(range(0,95,10))]
 tA,iouA=static_refine_xy(ren,Tcw,K,T,pairsA,R0,tA,bounds=.04)
 # ---------- static C: upright on floor after first placement ----------
 # These are stable, unoccluded observations from the visually verified rest interval.
 fitC=[290,310,330,350,370,390,410]
 objC={};perC={}
 with h5py.File(root+'/masks.h5','r') as f:
  for i in fitC:
   objC[i]=np.array(f['object'][i]);perC[i]=np.array(f['person'][i])
 # Fixed crop encloses every phase-C basket mask and render while remaining small enough to resolve the object.
 xc0,yc0,szc=690,710,105;Krc=crop_K(K,xc0,yc0,szc);valsC=[resized_masks(objC[i],perC[i],xc0,yc0,szc) for i in fitC]
 imgsC={};capC=cv2.VideoCapture(root+'/video.mp4')
 for i in range(980):
  ok,im=capC.read();assert ok
  if i in fitC:imgsC[i]=cv2.resize(im[yc0:yc0+szc,xc0:xc0+szc],(RES,RES),interpolation=cv2.INTER_AREA)[:,:,::-1].astype(np.float32)/255.
 def costC(par):
  x,y,z,yaw=par;R=Rot.from_euler('y',yaw,degrees=True).as_matrix()@R0;P=np.eye(4,dtype=np.float32);P[:3,:3]=R;P[:3,3]=[x,y,z];poses=np.repeat((Tcw@P)[None],len(valsC),axis=0);mm,rgb,_=ren.render_batch(poses.astype(np.float32),Krc[None],np.array([[RES,RES]],np.float32));mm=mm.cpu().numpy();rgb=rgb.cpu().numpy();ious=[];ccs=[]
  for j,(o,p) in enumerate(valsC):
   ious.append(mask_iou(mm[j],o,p));msk=o&(rgb[j].max(2)>.05)&~p;a=imgsC[fitC[j]][msk].mean(1);b=rgb[j][msk].mean(1);ccs.append(np.corrcoef(a,b)[0,1] if msk.sum()>20 and a.std()>1e-4 and b.std()>1e-4 else 0)
  pd=((ren.hull@R.T)@n).min()+np.dot(n,[x,y,z])+d;return 1-np.mean(ious)-.35*np.mean(ccs)+3.0*max(0,abs(pd)-.045)
 rc=differential_evolution(costC,[(.05,.2),(1.02,1.14),(3.80,4.03),(-180,180)],seed=4,popsize=6,maxiter=15,tol=.003,polish=False,disp=False);rrc=minimize(costC,rc.x,method='Nelder-Mead',options={'maxiter':120});xC,yC,zC,yawC=rrc.x;RC=Rot.from_euler('y',yawC,degrees=True).as_matrix()@R0;tC=np.array([xC,yC,zC],np.float32);iouC=1-rrc.fun
 # ---------- dynamic orientations ----------
 RB=robust_orientations(T,95,258,start_R=R0,end_R=RC)
 RD=robust_orientations(T,415,745,start_R=RC)
 RB=anchor_ramp(RB,start_R=R0,end_R=RC,n=12)
 RD=anchor_ramp(RD,start_R=RC,n=12)
 # The baseline flips to the opposite orientation branch during the final placement; bridge the last 12 frames smoothly to the static pose instead of accepting a one-frame 138 degree jump.
 nbridge=12
 for k in range(len(RB)-nbridge,len(RB)):
  sfrac=(k-(len(RB)-nbridge))/(nbridge-1);sfrac=sfrac*sfrac*(3-2*sfrac);RB[k]=matrix_slerp(RB[len(RB)-nbridge],RC,sfrac)
 # hand frames and final phase fit
 HR=hand_frames(r['human_keypoints_world'],np.arange(21,42));HL=hand_frames(r['human_keypoints_world'],np.arange(42,63))
 Hfin=HL
 # final fit: upright rotation, translation constrained near floor using visible frames.
 # Frames 849-861 are the stable part of the short 847-867 window in which SAM still sees the released basket.
  # Fixed 180 px crop centered on the visible final basket; chosen by eye from frames 847-867.
 fitframes=list(range(849,862));fitpairs=[(i,objm[i],perm[i]) for i in fitframes];x0,y0,sz,Kr=835,790,180,crop_K(K,835,790,180);vals=[]
 for i,o,p in fitpairs:vals.append((*resized_masks(o,p,x0,y0,sz),))
 # actual color crops for texture term
 imgs={};cap=cv2.VideoCapture(root+'/video.mp4')
 for i in range(980):
  ok,im=cap.read();assert ok
  if i in fitframes:imgs[i]=cv2.resize(im[y0:y0+sz,x0:x0+sz],(RES,RES),interpolation=cv2.INTER_AREA)[:,:,::-1].astype(np.float32)/255.
 def final_cost(par):
  x,y,z,yaw=par;R=Rot.from_euler('y',yaw,degrees=True).as_matrix()@R0;P=np.eye(4,dtype=np.float32);P[:3,:3]=R;P[:3,3]=[x,y,z];poses=np.repeat((Tcw@P)[None],len(vals),axis=0);mm,rgb,_=ren.render_batch(poses.astype(np.float32),Kr[None],np.array([[RES,RES]],np.float32));mm=mm.cpu().numpy();rgb=rgb.cpu().numpy()
  ious=[];ccs=[]
  for j,(o,p) in enumerate(vals):
   ious.append(mask_iou(mm[j],o,p));msk=o&(rgb[j].max(2)>.05)&~p
   if msk.sum()>20:
    a=imgs[fitframes[j]][msk].mean(1);b=rgb[j][msk].mean(1);ccs.append(np.corrcoef(a,b)[0,1] if a.std()>1e-4 and b.std()>1e-4 else 0)
   else:ccs.append(0)
  pd=((ren.hull@R.T)@n).min()+np.dot(n,[x,y,z])+d
  return 1-np.mean(ious)-.35*np.mean(ccs)+3.0*max(0,abs(pd)-.045)
 res=differential_evolution(final_cost,[(.62,.84),(1.07,1.18),(2.76,3.02),(-180,180)],seed=3,popsize=6,maxiter=13,tol=.005,polish=False,disp=False);rf=minimize(final_cost,res.x,method='Nelder-Mead',options={'maxiter':110});xE,yE,zE,yawE=rf.x;RE=Rot.from_euler('y',yawE,degrees=True).as_matrix()@R0;tE=np.array([xE,yE,zE],np.float32);print('static',tA,iouA,RC,tC,iouC,'final',rf.x,1-rf.fun,flush=True)
 # dynamic translations
 tB=np.zeros((164,3),np.float32);tD=np.zeros((331,3),np.float32)
 for j,i in enumerate(range(95,259)):tB[j]=align_xy(ren,Tcw,K,objm[i],perm[i],RB[j],T[i,:3,3])
 for j,i in enumerate(range(415,746)):tD[j]=align_xy(ren,Tcw,K,objm[i],perm[i],RD[j],T[i,:3,3])
 # Light temporal smoothing removes per-frame mask-centroid jitter; hard anchors keep phase transitions continuous.
 tB=gaussian_filter1d(tB,1.0,axis=0);tD=gaussian_filter1d(tD,1.0,axis=0)
 tB=vec_anchor_ramp(tB,start=tA,end=tC,n=10);tD=vec_anchor_ramp(tD,start=tC,n=10)
 # hand-anchored translation in occlusion, calibrated from visible carried poses
 HRv=HR;HLv=HL;offs=[]
 for i in range(580,731):
  if i not in objm:continue
  # left hand is closest in this interval
  h=r['human_keypoints_world'][i,42:63].mean(0);tt=tD[i-415];offs.append(HLv[i].T@(tt-h))
 off=np.median(np.array(offs),axis=0)
 pred=[]
 for i in range(746,846):pred.append(r['human_keypoints_world'][i,42:63].mean(0)+HLv[i]@off)
 pred=np.array(pred);pred=gaussian_filter1d(pred,2.0,axis=0);pred=vec_anchor_ramp(pred,start=tD[-1],n=10)
 # blend release into final static translation
 for k in range(35):pred[65+k]=(1-(k+1)/36)*pred[65+k]+((k+1)/36)*tE
 # orientation during occlusion: hand-propagated, eased toward final
 qstart=Rot.from_matrix(RD[745-415]).as_quat();qend=Rot.from_matrix(RE).as_quat();occR=[]
 for k,i in enumerate(range(746,846)):
  s=(i-745)/101.;hs=Rot.from_matrix(HLv[i]@HLv[745].T).as_quat();qp=quat_slerp(qstart,qstart,0);qp=Rot.from_quat(hs).as_matrix()@RD[745-415];q=matrix_slerp(qp,RE,s*s);occR.append(q)
 occR=np.array(occR,np.float32)
 # assemble
 Tout=T.copy();Tout[0:95]=np.tile(np.eye(4),(95,1,1));Tout[0:95,:3,:3]=R0;Tout[0:95,:3,3]=tA
 Tout[95:259,:3,:3]=RB;Tout[95:259,:3,3]=tB
 Tout[259:415]=np.tile(np.eye(4),(156,1,1));Tout[259:415,:3,:3]=RC;Tout[259:415,:3,3]=tC
 Tout[415:746,:3,:3]=RD;Tout[415:746,:3,3]=tD
 Tout[746:846,:3,:3]=occR;Tout[746:846,:3,3]=pred
 Tout[846:]=np.tile(np.eye(4),(134,1,1));Tout[846:,:3,:3]=RE;Tout[846:,:3,3]=tE
 np.savez(args.out,T_world_mesh=Tout.astype(np.float64))
 print('wrote',args.out,'frames',len(Tout),'runtime_s',round(time.time()-tstart,1),flush=True)
if __name__=='__main__':main()
