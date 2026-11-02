#!/usr/bin/env python3
"""Conservative deterministic image-evidence refinement for the supplied episode.

Run offline: python solve.py --inputs /work/inputs --out object_poses_world.npz
No trajectory cache, scratch files, downloaded data, or model weights are read.
All thresholds below are explicit reproducible algorithm constants.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import cv2, h5py, torch, trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.spatial.transform import Rotation

class Renderer:
 def __init__(self,mesh_path,K,size=160):
  self.device='cuda';self.K=np.asarray(K);self.size=size
  mesh=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
  mesh.merge_vertices(merge_tex=False,merge_norm=True)
  self.mesh=mesh; self.v=torch.as_tensor(np.asarray(mesh.vertices).copy(),device=self.device,dtype=torch.float32)
  self.faces=torch.as_tensor(np.asarray(mesh.faces).copy(),device=self.device,dtype=torch.int32)
  uv=np.asarray(mesh.visual.uv).copy();uv[:,1]=1-uv[:,1]
  self.uv=torch.as_tensor(uv,device=self.device,dtype=torch.float32)[None]
  tex=np.asarray(mesh.visual.material.baseColorTexture.convert('RGB')).copy()/255.
  self.tex=torch.as_tensor(tex,device=self.device,dtype=torch.float32)[None].contiguous()
  self.ctx=dr.RasterizeCudaContext()
  # Smooth semantic label: cyan body below the cap; cap begins near z=0.10 m.
  self.attr=torch.stack([self.v[:,2],torch.clamp((self.v[:,2]-.097)/.012,0,1)],-1)[None].contiguous()
 def render(self,R,t,boxes,texture=False,antialias=False):
  if not torch.is_tensor(R):R=torch.as_tensor(R,device=self.device,dtype=torch.float32)
  if not torch.is_tensor(t):t=torch.as_tensor(t,device=self.device,dtype=torch.float32)
  boxes=torch.as_tensor(boxes,device=self.device,dtype=torch.float32)
  cam=self.v[None]@R.transpose(1,2)+t[:,None,:]
  x,y,z=cam.unbind(-1);x0,y0,scale=boxes.unbind(-1);S=self.size
  xx=2*(self.K[0,0]*x+(self.K[0,2]-x0[:,None])*z)/(scale[:,None]*S)-z
  yy=2*(self.K[1,1]*y+(self.K[1,2]-y0[:,None])*z)/(scale[:,None]*S)-z
  clip=torch.stack([xx,yy,1.00002*z-.002,z],-1).contiguous()
  rast,_=dr.rasterize(self.ctx,clip,self.faces,resolution=[S,S])
  mask=(rast[...,3:] > 0).float()
  attr,_=dr.interpolate(self.attr,rast,self.faces)
  if antialias: mask=dr.antialias(mask, rast,clip,self.faces)
  color=None
  if texture:
   uv,_=dr.interpolate(self.uv,rast,self.faces);color=dr.texture(self.tex,uv,filter_mode='linear')*mask
  return mask[...,0],attr,color

def make_data(inputs,ids,K,T,size=160):
    cap=cv2.VideoCapture(str(inputs/'video.mp4')); h=h5py.File(inputs/'masks.h5'); boxes=[];images=[];masks=[];people=[];ids=set(map(int,ids));frame=0;times=time.time()
    while cap.isOpened():
        ok,im=cap.read()
        if not ok:break
        if frame in ids:
            uv=K@T[frame,:3,3];uv=uv[:2]/uv[2]
            x,y=np.round(uv-size/2).astype(int);x=int(np.clip(x,0,im.shape[1]-size));y=int(np.clip(y,0,im.shape[0]-size))
            images.append(im[y:y+size,x:x+size,::-1].copy());masks.append(h['object'][frame,y:y+size,x:x+size]);people.append(h['person'][frame,y:y+size,x:x+size]);boxes.append([x,y,1.])
        frame+=1
    cap.release();h.close()
    images=np.stack(images);masks=np.stack(masks);people=np.stack(people);boxes=np.array(boxes)
    # Ignore people outside the provided object segmentation, permitting occlusion.
    valid=(~people)|masks
    # Material-color cues are used only where the released object mask is present.
    rgb=images.astype(float)/255; blue=(rgb[...,2]>1.15*rgb[...,1])&(rgb[...,2]>1.45*rgb[...,0])&(rgb[...,2]>.18)&masks
    cyan=(rgb[...,1]>1.22*rgb[...,0])&(rgb[...,2]>.92*rgb[...,0])&(rgb[...,1]>.84*rgb[...,2])&(rgb[...,1]>.23)&masks
    # Soft body probability from color; cap pixels are darker / non-cyan.
    nonbody=masks&(~cyan)
    dist=np.stack([distance_transform_edt(~m)/12 for m in masks]).clip(0,2).astype('float32')
    print('loaded crops',len(boxes),'seconds',round(time.time()-times,2),flush=True)
    return {'ids':np.array(sorted(ids)),'boxes':boxes,'images':images,'masks':masks,'valid':valid,'blue':blue,'cyan':cyan,'nonbody':nonbody,'dist':dist}

def to_torch(data):
    return {k:torch.as_tensor(data[k],device='cuda',dtype=torch.float32) for k in ['boxes','images','masks','valid','blue','cyan','nonbody','dist']}

def losses(renderer,R,t,d,texture=False):
    m,attr,col=renderer.render(R,t,d['boxes'],texture=texture,antialias=True)
    obs=d['masks'];v=d['valid'];area=obs.sum((1,2)).clamp(min=100)
    inter=(m*obs).sum((1,2));union=(m*v).sum((1,2))+area-inter
    iou=inter/union.clamp(min=1)
    outline=(m*d['dist']*v).sum((1,2))/area
    # Cap/body discrepancy, positive only on confidently cyan observed pixels.
    cap=torch.clamp((attr[...,0]-.108)/.010,0,1)*m
    cap_wrong=(cap*d['cyan']).sum((1,2))/area
    # The blue latch occupies the top half of the mesh.
    blue_wrong=(torch.clamp((.035-attr[...,0])/.06,0,1)*m*d['blue']).sum((1,2))/area
    l=1-iou+.12*outline+.9*cap_wrong+2*blue_wrong
    if texture:
        # Color-invariant blue ratio gives a stable asymmetric feature.
        br=torch.sigmoid((col[...,2]-1.16*col[...,1]-.015)*30)*torch.sigmoid((col[...,2]-1.4*col[...,0])*30)*m
        ob=d['blue'];both=(m*obs); n=ob.sum((1,2));
        berror=((br-ob).abs()*both).sum((1,2))/area
        blue_union=(br*v+ob-br*ob).sum((1,2)).clamp(min=20)
        blue_inter=(br*ob).sum((1,2))
        l=l+.50*(1-blue_inter/blue_union)*(n>12)
    return l,torch.stack([iou,cap_wrong,blue_wrong,outline],1),m


@torch.no_grad()
def search(renderer,Rs,ts,d,rot_reg=.012,shift_reg=.004,rounds=3):
    # Forward-only rasterization avoids the nondeterministic atomic reductions
    # used in differentiable rendering. Search arithmetic and ordering are fixed.
    Rs=np.asarray(Rs,dtype=np.float64).copy();ts=np.asarray(ts,dtype=np.float64).copy();R0=Rs.copy();t0=ts.copy();N=len(Rs);K=renderer.K
    uv0=t0[:,:2]/t0[:,2:]*[K[0,0],K[1,1]]+K[:2,2];uv=uv0.copy();rad=np.linalg.norm(t0,axis=-1)
    def trans(u):
        ray=np.concatenate([(u-K[:2,2])/[K[0,0],K[1,1]],np.ones((len(u),1))],axis=1);return ray/np.linalg.norm(ray,axis=1,keepdims=True)*np.repeat(rad,len(u)//N)[:,None]
    def evaluate(R,t,u,dd):
        l,st,_=losses(renderer,R,t,dd,texture=True);l=l.cpu().numpy();nn=len(R)//N
        theta=np.arccos(np.clip((np.einsum('nij,nij->n',R,np.repeat(R0,nn,axis=0))-1)/2,-1,1));shift=u-np.repeat(uv0,nn,axis=0)
        return l+rot_reg*theta**2+shift_reg*np.sum((shift/4)**2,axis=1),st.cpu().numpy()
    best,stats=evaluate(Rs,ts,uv,d);dd={k:v.repeat_interleave(3,0) for k,v in d.items()}
    axes=np.eye(3)
    for angle,pix in [(12,3),(6,1.5),(3,.75),(1.5,.375),(.75,.1875)]:
        for cycle in range(rounds):
            changed=0
            for dim in range(5):
                rc=np.repeat(Rs[:,None],3,axis=1);uc=np.repeat(uv[:,None],3,axis=1)
                if dim<3:
                    deltas=Rotation.from_rotvec(np.array([0,-1,1])[:,None]*axes[dim]*np.deg2rad(angle)).as_matrix();rc=deltas[None]@rc
                else:uc[:,:,dim-3]+=np.array([0,-pix,pix])
                uc=np.clip(uc,uv0[:,None]-16,uv0[:,None]+16);tc=trans(uc.reshape(-1,2));rc=rc.reshape(-1,3,3);uc=uc.reshape(-1,2)
                cost,st=evaluate(rc,tc,uc,dd);cost=cost.reshape(N,3);ch=np.argmin(cost,axis=1);ind=np.arange(N)*3+ch
                Rs=rc[ind];ts=tc[ind];uv=uc[ind];best=cost[np.arange(N),ch];stats=st[ind];changed+=np.count_nonzero(ch)
            if changed==0:break
    return Rs,ts,best,stats

def generate(inputs):
 torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4);cv2.setNumThreads(1);c=json.load(open(inputs/'camera.json'));K=np.array(c['K']);a=np.load(inputs/'reconstruction.npz');T=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];ids=np.arange(len(T));data=make_data(inputs,ids,K,T);r=Renderer(inputs/'object_mesh.glb',K);d=to_torch(data);F=len(T);N=9
 Rall=np.zeros((F,N,3,3));tall=np.zeros((F,N,3));score=np.zeros((F,N));stats=np.zeros((F,N,4));Rall[:,0]=T[:,:3,:3];tall[:,0]=T[:,:3,3];start=time.time()
 for j in range(0,F,4):
  ii=ids[j:j+4];n=len(ii);R0=np.stack([T[ii,:3,:3],a['T_cam_mesh_foundationpose'][ii,:3,:3],T[np.clip(ii-10,0,F-1),:3,:3],T[np.clip(ii+10,0,F-1),:3,:3],T[ii,:3,:3]@Rotation.from_euler('z',90,degrees=True).as_matrix(),T[ii,:3,:3]@Rotation.from_euler('z',-90,degrees=True).as_matrix(),T[ii,:3,:3]@Rotation.from_euler('z',180,degrees=True).as_matrix(),a['T_cam_mesh_foundationpose'][ii,:3,:3]@Rotation.from_euler('x',180,degrees=True).as_matrix()],axis=1)
  t0=np.repeat(T[ii,None,:3,3],N-1,axis=1);dd={k:v[j:j+n].repeat_interleave(N-1,0) for k,v in d.items()}
  RR,tt,ss,st=search(r,R0.reshape(-1,3,3),t0.reshape(-1,3),dd,rot_reg=.012,shift_reg=.004)
  Rall[ii,1:]=RR.reshape(n,N-1,3,3);tall[ii,1:]=tt.reshape(n,N-1,3);
  dd={k:v[j:j+n].repeat_interleave(N,0) for k,v in d.items()}
  with torch.no_grad():l,s,_=losses(r,Rall[ii].reshape(-1,3,3),tall[ii].reshape(-1,3),dd,texture=True)
  score[ii]=l.cpu().numpy().reshape(n,N);stats[ii]=s.cpu().numpy().reshape(n,N,4)
  if j%40==0:print('candidates',j,'/',F,'seconds',round(time.time()-start,1),flush=True)
 print('base search seconds',time.time()-start,flush=True);return dict(R=Rall,t=tall,scores=score,stats=stats,**data),r


@torch.no_grad()
def compute_blue(c,renderer):
 R=c['R'];t=c['t'];F,N=R.shape[:2];score=np.zeros((F,N))
 for j in range(0,F,4):
  n=min(4,F-j);m,at,col=renderer.render(R[j:j+n].reshape(-1,3,3),t[j:j+n].reshape(-1,3),np.repeat(c['boxes'][j:j+n],N,axis=0),texture=True,antialias=True)
  br=torch.sigmoid((col[...,2]-1.16*col[...,1]-.015)*30)*torch.sigmoid((col[...,2]-1.4*col[...,0])*30)*m
  ob=torch.as_tensor(np.repeat(c['blue'][j:j+n],N,axis=0),device='cuda',dtype=torch.float32);v=torch.as_tensor(np.repeat(c['valid'][j:j+n],N,axis=0),device='cuda',dtype=torch.float32)
  val=(br*ob).sum((1,2))/(br*v+ob-br*ob).sum((1,2)).clamp(min=20);score[j:j+n]=val.cpu().numpy().reshape(n,N)
 return score

def add_roll_candidates(c,renderer):
 R=c['R'];t=c['t'];F,N=R.shape[:2];blue=compute_blue(c,renderer);c['blue_iou']=blue;count=c['blue'].sum((1,2));
 ids=np.where((count>80)&(blue[:,0]<.20)&(c['stats'][:,:,0].max(1)>.70))[0];Nadd=8
 RR=np.repeat(R[:,0,None],Nadd,axis=1);tt=np.repeat(t[:,0,None],Nadd,axis=1);ss=np.repeat(c['scores'][:,0,None],Nadd,axis=1);st=np.repeat(c['stats'][:,0,None],Nadd,axis=1);d=to_torch(c);start=time.time();print('axial search frames',len(ids),flush=True)
 for j in range(0,len(ids),4):
  ii=ids[j:j+4];n=len(ii);best=np.argmax(c['stats'][ii,:,0]-.2*c['stats'][ii,:,2],axis=1);seed=R[ii,best];ts=t[ii,best]
  delta=Rotation.from_euler('z',np.arange(8)*45,degrees=True).as_matrix();R0=seed[:,None]@delta[None];t0=np.repeat(ts[:,None],Nadd,axis=1);dd={k:v[ii].repeat_interleave(Nadd,0) for k,v in d.items()}
  r1,t1,s1,st1=search(renderer,R0.reshape(-1,3,3),t0.reshape(-1,3),dd,rot_reg=.012,shift_reg=.004,rounds=2)
  with torch.no_grad():l,stats,_=losses(renderer,r1,t1,dd,texture=True)
  RR[ii]=r1.reshape(n,Nadd,3,3);tt[ii]=t1.reshape(n,Nadd,3);ss[ii]=l.cpu().numpy().reshape(n,Nadd);st[ii]=stats.cpu().numpy().reshape(n,Nadd,4)
 for name,val in [('R',RR),('t',tt),('scores',ss),('stats',st)]:c[name]=np.concatenate([c[name],val],axis=1)
 c['blue_iou']=compute_blue(c,renderer);print('axial search seconds',time.time()-start,flush=True);return c


def angle(A,B):
 return np.arccos(np.clip((np.einsum('...ij,...ij->...',A,B)-1)/2,-1,1))

def choose(c,K):
 R=c['R'];t=c['t'];l=c['scores'];stats=c['stats'];masks=c['masks'];F,N=l.shape
 uv=t[...,:2]/t[...,2:]*[K[0,0],K[1,1]]+K[:2,2];ang=angle(R,R[:,0,None]);axis=np.arccos(np.clip(np.einsum('fni,fi->fn',R[:,:,:,2],R[:,0,:,2]),-1,1));shift=np.linalg.norm(uv-uv[:,0,None],axis=-1)
 pca=np.zeros(F);elong=np.zeros(F)
 for f in range(F):
  y,x=np.where(masks[f]);cov=np.cov(np.stack([x,y]));val,vec=np.linalg.eigh(cov);a=vec[:,-1];p=R[f,0,:,2][None]*np.array([-.1,.1])[:,None]+t[f,0];q=p@K.T;q=q[:,:2]/q[:,2:];d=q[1]-q[0];pca[f]=np.degrees(np.arccos(np.clip(abs(a@d)/np.linalg.norm(d),0,1)));elong[f]=val[1]/max(val[0],1)
 # Silhouette size alone cannot establish out-of-plane tilt at fixed distance.
 # Require an independent projected-axis error, or gross silhouette failure.
 evidence=((pca>12)&(elong>2.2))|((stats[:,0,0]<.55)&(elong>2.2))
 if 'blue_iou' in c:
  bi=c['blue_iou'];bcount=c['blue'].sum((1,2));color_evidence=(bcount>80)&(bi[:,0]<.15)&np.any((bi>.50)&(stats[:,:,0]>.72),axis=1)
  evidence|=color_evidence
 eligible=binary_dilation(evidence,iterations=3)
 gain=l[:,0,None]-l
 valid=(gain>.10)&(stats[:,:,0]>.63)&(stats[:,:,0]>stats[:,0,None,0]+.05)&eligible[:,None]
 valid&=(axis<np.deg2rad(110))|((gain>.40)&(stats[:,:,0]>.82))
 if 'blue_iou' in c:
  # A plainly visible latch rules out rotations that hide it. The silhouette
  # cannot override this independent material cue for an almost round bottle.
  strong=(bcount>80)&np.any((bi>.50)&(stats[:,:,0]>.72),axis=1)
  threshold=np.maximum(.35,np.max(np.where(stats[:,:,0]>.72,bi,0),axis=1)-.25)
  valid&=(~strong[:,None])|(bi>=threshold[:,None])
 valid[:,0]=True
 unary=l+.045*np.minimum(ang**2,5)+.010*(shift/3)**2
 unary=np.where(valid,unary,1e4)
 dp=unary[0].copy();back=np.zeros((F,N),int)
 for i in range(1,F):
  rot=angle(R[i-1,:,None],R[i,None]);ax=np.arccos(np.clip(R[i-1,:,:,2]@R[i,:,:,2].T,-1,1))
  # Cap temporal penalties: an uncorrectable isolated frame must not prevent
  # correction of a neighboring frame with decisive independent image evidence.
  temporal=.10*np.minimum((rot/.4)**2,3)+.08*np.minimum((ax/.3)**2,3)
  du=uv[i,None]-uv[i-1,:,None];base=uv[i,0]-uv[i-1,0];temporal+=.002*np.minimum(np.sum((du-base)**2,axis=-1)/9,20)
  cost=dp[:,None]+temporal;back[i]=np.argmin(cost,axis=0);dp=unary[i]+cost.min(0)
 chosen=np.zeros(F,int);chosen[-1]=np.argmin(dp)
 for i in range(F-1,0,-1):chosen[i-1]=back[i,chosen[i]]
 return chosen,dict(gain=gain,angle=ang,axis=axis,shift=shift,pca=pca,elong=elong,eligible=eligible)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.out.parent.mkdir(parents=True,exist_ok=True)
    camera=json.loads((args.inputs/'camera.json').read_text())
    K=np.array(camera['K'],dtype=np.float64)
    world_cam=np.array(camera['T_world_cam'],dtype=np.float64)
    with np.load(args.inputs/'reconstruction.npz') as data:
        baseline=data['T_world_mesh'].astype(np.float64)
    expected=json.loads((args.inputs/'episode.json').read_text())['frames']
    assert len(baseline)==expected
    candidates,renderer=generate(args.inputs)
    candidates=add_roll_candidates(candidates,renderer)
    chosen,meta=choose(candidates,K)
    ii=np.arange(len(chosen));changed=chosen!=0
    result=baseline.copy();cam_base=np.linalg.inv(world_cam)@baseline
    selected_R=candidates['R'][ii,chosen].copy()
    selected_t=candidates['t'][ii,chosen].copy()
    # Preserve calibrated Euclidean camera-to-centre distance in float64.
    radii=np.linalg.norm(cam_base[:,:3,3],axis=1)
    selected_t*=radii[:,None]/np.linalg.norm(selected_t,axis=1,keepdims=True)
    if changed.any():
        # Repair tiny floating-point nonorthogonality only in edited rotations.
        # Unchanged output entries retain the original baseline values exactly.
        U,_,Vh=np.linalg.svd(selected_R[changed]);rr=U@Vh
        correction=np.ones((len(rr),3));correction[:,2]=np.linalg.det(rr)
        rr=(U*correction[:,None,:])@Vh
        ct=np.tile(np.eye(4),(changed.sum(),1,1))
        ct[:,:3,:3]=rr;ct[:,:3,3]=selected_t[changed]
        result[changed]=world_cam@ct
    result=result.astype(np.float64)
    assert result.shape==(expected,4,4) and np.isfinite(result).all()
    assert np.array_equal(result[~changed],baseline[~changed])
    assert np.max(np.abs(result[:,3,:]-[0,0,0,1]))<1e-12
    distance=np.linalg.norm((np.linalg.inv(world_cam)@result)[:,:3,3],axis=1)
    assert np.max(np.abs(distance-radii))<1e-12
    assert np.max(np.abs(np.linalg.det(result[:,:3,:3])-1))<1e-5
    np.savez_compressed(args.out,T_world_mesh=result)
    base_iou=candidates['stats'][:,0,0];final_iou=candidates['stats'][ii,chosen,0]
    blue=candidates['blue_iou'];blue_visible=candidates['blue'].sum((1,2))>80
    intervals=np.where(np.diff(np.r_[False,changed,False]))[0].reshape(-1,2)
    def pct(v):return np.percentile(v,[0,25,50,75,95,100]).tolist()
    metrics=dict(frames=expected,changed_frames=int(changed.sum()),unchanged_frames=int((~changed).sum()),
        changed_intervals_inclusive=[[int(a),int(b-1)] for a,b in intervals],
        silhouette_iou_all_before=float(base_iou.mean()),silhouette_iou_all_after=float(final_iou.mean()),
        silhouette_iou_changed_before=float(base_iou[changed].mean()),silhouette_iou_changed_after=float(final_iou[changed].mean()),
        visible_blue_iou_before=float(blue[blue_visible,0].mean()),visible_blue_iou_after=float(blue[ii,chosen][blue_visible].mean()),
        rotation_change_degrees_percentiles=pct(np.degrees(meta['angle'][ii,chosen][changed])),
        pixel_shift_percentiles=pct(meta['shift'][ii,chosen][changed]),
        translation_change_m_percentiles=pct(np.linalg.norm(result[changed,:3,3]-baseline[changed,:3,3],axis=1)),
        max_camera_distance_change_m=float(np.max(np.abs(distance-radii))),
        max_rotation_orthogonality_error=float(np.max(np.abs(result[:,:3,:3].transpose(0,2,1)@result[:,:3,:3]-np.eye(3)))),
        run_seconds=time.perf_counter()-started,ground_truth_used=False)
    (args.out.parent/'metrics.json').write_text(json.dumps(metrics,indent=2))
    with open(args.out.parent/'frame_diagnostics.csv','w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['frame','changed','candidate','baseline_iou','final_iou','baseline_blue_iou','final_blue_iou','rotation_change_deg','axis_change_deg','pixel_shift','baseline_axis_pca_error_deg','image_score_before','image_score_after'])
        for i,j in enumerate(chosen):
            writer.writerow([i,int(changed[i]),int(j),base_iou[i],final_iou[i],blue[i,0],blue[i,j],np.degrees(meta['angle'][i,j]),np.degrees(meta['axis'][i,j]),meta['shift'][i,j],meta['pca'][i],candidates['scores'][i,0],candidates['scores'][i,j]])
    print(json.dumps(metrics,indent=2))
    print('Saved',args.out,flush=True)

if __name__=='__main__':
    main()
