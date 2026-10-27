#!/usr/bin/env python3
"""Offline, deterministic, fixed-distance cane trajectory correction.

Fits only rotations and image-plane centre direction to supplied masks and the
orange/black texture. Uses no learned models, downloaded assets, or annotations.
The baseline per-frame camera distance is an exact constraint. See REPORT.md.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ.setdefault('OMP_NUM_THREADS', '4')
import nvdiffrast.torch as dr
import os,time,json,argparse
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,cv2,h5py,trimesh,torch
from scipy.spatial.transform import Rotation
torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
def prepare_mesh(path,n=1000):
 m=trimesh.load(path,force='scene',process=False).to_geometry();p,f=trimesh.sample.sample_surface(m,18000,seed=0);key=np.floor(p/.009).astype(int);_,ix=np.unique(key,axis=0,return_index=True);ix=np.sort(ix);rng=np.random.default_rng(0);ix=rng.choice(ix,min(n,len(ix)),replace=False);p=p[ix];f=f[ix]
 
 from scipy.spatial import cKDTree
 _,vi=cKDTree(m.vertices).query(p);uv=m.visual.uv[vi];tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));rgb=tex[np.clip(((1-uv[:,1])*(len(tex)-1)).astype(int),0,len(tex)-1),np.clip((uv[:,0]*(tex.shape[1]-1)).astype(int),0,tex.shape[1]-1)];hsv=cv2.cvtColor(rgb[None],cv2.COLOR_RGB2HSV)[0];orange=(hsv[:,0]<28)&(hsv[:,0]>3)&(hsv[:,1]>110)&(hsv[:,2]>70)
 return m,p,orange
def ortho6(x):
 a=torch.nn.functional.normalize(x[...,:3],dim=-1);b=x[...,3:];b=torch.nn.functional.normalize(b-(a*b).sum(-1,keepdim=True)*a,dim=-1);c=torch.cross(a,b,dim=-1);return torch.stack([a,b,c],dim=-1)
def sample_maps(maps, q):
 """Bilinear sampling of constant maps; no CUDA scatter in backward."""
 B,C,H,W=maps.shape
 x=(q[...,0]*(W-1)/1535.).clamp(0,W-1)
 y=(q[...,1]*(H-1)/1151.).clamp(0,H-1)
 x0=x.floor().long();y0=y.floor().long()
 x1=(x0+1).clamp(max=W-1);y1=(y0+1).clamp(max=H-1)
 dx=x-x0;dy=y-y0
 flat=maps.reshape(B,C,-1)
 def get(xx,yy):
  return flat.gather(2,(yy*W+xx)[:,None].expand(-1,C,-1))
 return (get(x0,y0)*(1-dx[:,None])*(1-dy[:,None])+
         get(x1,y0)*dx[:,None]*(1-dy[:,None])+
         get(x0,y1)*(1-dx[:,None])*dy[:,None]+
         get(x1,y1)*dx[:,None]*dy[:,None])
def project(R,uv,d,pts,K):
 ray=torch.cat([(uv-K[None,:2,2])/torch.tensor([K[0,0],K[1,1]],device=uv.device),torch.ones_like(uv[...,:1])],-1);t=torch.nn.functional.normalize(ray,dim=-1)*d[...,None];q=torch.einsum('bij,nj->bni',R,pts)+t[:,None];return q[...,:2]/q[...,2:]*torch.tensor([K[0,0],K[1,1]],device=uv.device)+K[None,:2,2],t
def fit_candidates(indices,out,inputs,steps=220,n=800,seeds=None):
 start=time.time();p=inputs.rstrip('/')+'/';a=np.load(p+'reconstruction.npz');c=json.load(open(p+'camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);base=np.linalg.inv(W)[None]@a['T_world_mesh'];m,pts,orange=prepare_mesh(p+'object_mesh.glb',n);h=h5py.File(p+'masks.h5');device='cuda';
 allres=[];allstats=[];allcandidates=[];allscores=[];video=cv2.VideoCapture(p+'video.mp4')
 for st in range(0,len(indices),8):
  ids=indices[st:st+8];Rs=[];uvs=[];ds=[];obs=[];maps=[];scales=[];meta=[];orangeobs=[];colorweights=[]
  for i in ids:
   mask=np.asarray(h['object'][i],np.uint8);person=np.asarray(h['person'][i],np.uint8);yx=np.argwhere(mask);xy=yx[:,::-1].astype(float);cov=np.cov(xy.T);ev,evec=np.linalg.eigh(cov);axis=evec[:,-1];
   # Rotate candidate long-axis image direction to observed mask PCA, with both signs.
   starts=[]
   for source in [base[i,:3,:3],a['T_cam_mesh_foundationpose'][i,:3,:3]]:
    for sign in [1,-1]:
     v=source[:,2]; ang=np.arctan2(sign*axis[1],sign*axis[0])-np.arctan2(v[1],v[0]);rz=Rotation.from_rotvec([0,0,ang]).as_matrix();rr=rz@source
     for zroll in [0,np.pi/2,np.pi,3*np.pi/2]:starts.append(rr@Rotation.from_rotvec([0,0,zroll]).as_matrix())
   starts=[base[i,:3,:3]]+starts
   if seeds is not None: starts=list(seeds[i])
   N=len(starts);yy,xx=np.where(mask);left=max(0,min(xx.min(),int(K[0,0]*base[i,0,3]/base[i,2,3]+K[0,2]))-190);top=max(0,min(yy.min(),int(K[1,1]*base[i,1,3]/base[i,2,3]+K[1,2]))-190);right=min(1536,max(xx.max(),int(K[0,0]*base[i,0,3]/base[i,2,3]+K[0,2]))+191);bottom=min(1152,max(yy.max(),int(K[1,1]*base[i,1,3]/base[i,2,3]+K[1,2]))+191)
   # Maps use full-frame quarter resolution; distance measured in original pixels.
   video.set(1,int(i));_,image=video.read();hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);dil=cv2.dilate(mask,np.ones((5,5),np.uint8));og=(hsv[:,:,0]>3)&(hsv[:,:,0]<28)&(hsv[:,:,1]>110)&(hsv[:,:,2]>70)&(dil>0);occluder=person*(1-dil);odt=cv2.distanceTransform((1-(og|occluder)).astype('uint8'),cv2.DIST_L2,5);odt=cv2.resize(odt,(384,288),interpolation=cv2.INTER_AREA);oyx=np.argwhere(og);colorweight=float(len(oyx)>=8);oyx=oyx if len(oyx) else yx;rng=np.random.default_rng(i+1000);oix=rng.choice(len(oyx),192,replace=len(oyx)<192);oob=oyx[oix,::-1].astype(float)
   merged=(mask|person);dt=cv2.distanceTransform(1-merged,cv2.DIST_L2,5);dt=cv2.resize(dt,(384,288),interpolation=cv2.INTER_AREA);pm=cv2.resize(occluder.astype(float),(384,288),interpolation=cv2.INTER_AREA)
   rng=np.random.default_rng(i);ix=rng.choice(len(xy),256,replace=len(xy)<256);ob=xy[ix];
   dist=np.linalg.norm(base[i,:3,3]);uv=K@base[i,:3,3];uv=uv[:2]/uv[2];
   Rs.extend(starts);uvs.extend([uv]*N);ds.extend([dist]*N);obs.extend([ob]*N);maps.extend([np.stack([dt,pm,odt])]*N);orangeobs.extend([oob]*N);colorweights.extend([colorweight]*N);scales.extend([np.sqrt(ev[-1])]*N);meta.append((i,N,len(xy)))
  rt=torch.tensor(np.array(Rs),dtype=torch.float32,device=device);x=torch.nn.Parameter(torch.cat([rt[:,:,0],rt[:,:,1]],dim=1));u0=torch.tensor(np.array(uvs),dtype=torch.float32,device=device);du=torch.nn.Parameter(torch.zeros_like(u0));d=torch.tensor(ds,dtype=torch.float32,device=device);ob=torch.tensor(np.array(obs),dtype=torch.float32,device=device);maps=torch.tensor(np.array(maps),dtype=torch.float32,device=device);pt=torch.tensor(pts,dtype=torch.float32,device=device);kt=torch.tensor(K,dtype=torch.float32,device=device);
  oob=torch.tensor(np.array(orangeobs),dtype=torch.float32,device=device);oi=torch.tensor(orange,device=device);cw=torch.tensor(colorweights,device=device)
  opt=torch.optim.Adam([{'params':[x],'lr':.025},{'params':[du],'lr':1.0}]);
  def loss():
   R=ortho6(x);q,t=project(R,u0+du,d,pt,kt);grid=(q/torch.tensor([1535.,1151.],device=device)*2-1)[:,None];sm=sample_maps(maps,q);dist=sm[:,0];vis=(1-sm[:,1]).clamp(0,1);fw=(torch.sqrt(dist**2+4)-2).sum(-1)/(vis.sum(-1)+80);
   D=torch.cdist(ob,q);bw=(torch.sqrt(D.min(-1).values**2+4)-2).mean(-1);
   ofw=(torch.sqrt(sm[:,2,oi]**2+4)-2).sum(-1)/((1-sm[:,1,oi]).sum(-1)+40);oD=torch.cdist(oob,q[:,oi]);obw=(torch.sqrt(oD.min(-1).values**2+4)-2).mean(-1);score=.5*fw+.7*bw+cw*(.8*ofw+1.2*obw);reg=.00015*(du**2).sum(-1);return score+reg,score,R,t,fw,bw
  for it in range(steps):
   opt.zero_grad();L,*_=loss();L.sum().backward();opt.step();
   if it==int(steps*.7):
    for pg in opt.param_groups:pg['lr']*=.3
  with torch.no_grad():L,s,R,t,fw,bw=loss()
  R=R.cpu().numpy();t=t.cpu().numpy();s=s.cpu().numpy();fw=fw.cpu().numpy();bw=bw.cpu().numpy();j=0
  for i,N,area in meta:
   allcandidates.append(np.concatenate([R[j:j+N],t[j:j+N,:,None]],-1));allscores.append(s[j:j+N]);order=np.argsort(s[j:j+N]);k=j+order[0];mat=np.eye(4);mat[:3,:3]=R[k];mat[:3,3]=t[k];allres.append(mat);allstats.append([i,s[k],fw[k],bw[k],order[0],area]);print(i,'score',np.round(s[k],2),'fw/bw',np.round([fw[k],bw[k]],2),'choice',order[:4].tolist(), 'top',np.round(s[j+order[:4]],2),'axis',np.round(R[k,:,2],2),flush=True);j+=N
  print('elapsed',round(time.time()-start,1),flush=True)
  np.savez(out,T_cam_mesh=np.array(allres),indices=indices[:len(allres)],stats=np.array(allstats),candidates=np.array(allcandidates),scores=np.array(allscores))
 np.savez(out,T_cam_mesh=np.array(allres),indices=indices,stats=np.array(allstats),candidates=np.array(allcandidates),scores=np.array(allscores));print('saved',out,'seconds',time.time()-start)

class Evaluator:
 def __init__(self,inputs='/work/inputs'):
  self.inputs=inputs;m=trimesh.load(inputs+'/object_mesh.glb',force='scene',process=False).to_geometry();m.merge_vertices();device='cuda';self.v=torch.tensor(np.array(m.vertices),dtype=torch.float32,device=device);self.faces=torch.tensor(m.faces.astype('int32'),device=device);self.uv=torch.tensor(np.array(m.visual.uv),dtype=torch.float32,device=device);self.tex=torch.tensor(np.array(m.visual.material.baseColorTexture.convert('RGB'))[::-1].copy()/255,dtype=torch.float32,device=device)[None];self.ctx=dr.RasterizeCudaContext();c=json.load(open(inputs+'/camera.json'));self.K=np.array(c['K']);self.kt=torch.tensor(self.K,dtype=torch.float32,device=device);self.vid=cv2.VideoCapture(inputs+'/video.mp4');self.h=h5py.File(inputs+'/masks.h5');a=np.load(inputs+'/reconstruction.npz');self.base=np.linalg.inv(c['T_world_cam'])[None]@a['T_world_mesh']
 def frame(self,i,poses,save=None):
  self.vid.set(1,int(i));_,im=self.vid.read();mask=np.array(self.h['object'][i],np.uint8);person=np.array(self.h['person'][i],np.uint8);
  if not np.any(mask): return np.zeros((len(poses),7),dtype=np.float64)
  yy,xx=np.where(mask);size=max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)+80;cx=(xx.max()+xx.min())/2;cy=(yy.max()+yy.min())/2;scale=320/size;x0=cx-size/2;y0=cy-size/2;A=np.array([[scale,0,-x0*scale],[0,scale,-y0*scale]]);S=320
  obs=cv2.warpAffine(mask,A,(S,S),flags=cv2.INTER_NEAREST);occ=cv2.warpAffine(person,A,(S,S),flags=cv2.INTER_NEAREST);image=cv2.warpAffine(im,A,(S,S));hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);dil=cv2.dilate(obs,np.ones((3,3),np.uint8));orange=((hsv[:,:,0]>3)&(hsv[:,:,0]<28)&(hsv[:,:,1]>110)&(hsv[:,:,2]>70)&(dil>0)).astype('uint8');occ=occ*(1-dil);dt=cv2.distanceTransform(1-obs,cv2.DIST_L2,5);odt=cv2.distanceTransform(1-orange,cv2.DIST_L2,5)
  predictions=[];colors=[]
  # Bound peak GPU memory independently of the number of candidate poses.
  for chunk in range(0,len(poses),3):
   part=poses[chunk:chunk+3]
   Rs=torch.tensor(part[:,:3,:3],dtype=torch.float32,device='cuda');ts=torch.tensor(part[:,:3,3],dtype=torch.float32,device='cuda')
   with torch.no_grad():
    q=torch.einsum('bij,nj->bni',Rs,self.v)+ts[:,None]
    xy=q@self.kt.T;xy=xy[...,:2]/xy[...,2:]
    xy=(xy-torch.tensor([x0,y0],device='cuda',dtype=torch.float32))*scale
    z=q[...,2];near=.1;far=10.
    clip=torch.stack([(2*xy[...,0]/S-1)*z,(2*xy[...,1]/S-1)*z,((far+near)/(far-near))*z-2*far*near/(far-near),z],-1)
    rast,_=dr.rasterize(self.ctx,clip,self.faces,[S,S])
    uv,_=dr.interpolate(self.uv[None],rast,self.faces)
    rgbpart=dr.texture(self.tex,uv,filter_mode='linear')
    predictions.append((rast[...,3]>0).cpu().numpy())
    colors.append((rgbpart.cpu().numpy()*255).astype('uint8'))
  pred=np.concatenate(predictions);rgb=np.concatenate(colors)
  vals=[];tiles=[]
  for j in range(len(poses)):
   pr=pred[j].astype('uint8');vis=pr*(1-occ);pdt=cv2.distanceTransform(1-pr,cv2.DIST_L2,5);rh=cv2.cvtColor(rgb[j],cv2.COLOR_RGB2HSV);po=((rh[:,:,0]>3)&(rh[:,:,0]<28)&(rh[:,:,1]>90)&(rh[:,:,2]>65)&(pr>0)).astype('uint8');podt=cv2.distanceTransform(1-po,cv2.DIST_L2,5)
   def mean(v,m):return float(np.sum(v*m)/max(1,np.sum(m)))
   # Distances have a 1 px noise allowance in source-image units, then robust linear tails.
   def robust(x):return np.sqrt((np.minimum(x,1e4)/scale)**2+1)-1
   fw=mean(robust(dt),vis);bw=mean(robust(pdt),obs);ofw=mean(robust(odt),po*(1-occ));obw=mean(robust(podt),orange);intersection=np.sum(vis*obs);iou=intersection/max(1,np.sum(vis|obs));cost=.45*fw+.65*bw+( .65*ofw+.9*obw if np.any(orange) else 0.);vals.append([cost,fw,bw,ofw,obw,iou,float(vis.sum()/max(1,pr.sum()))])
   if save:
    rr=rgb[j][:,:,::-1].copy();rr[pr==0]=245;combo=np.concatenate([image,rr],1);cv2.putText(combo,f'{i} cand {j}: {cost:.2f} IoU {iou:.2f}',(4,20),0,.5,(10,40,255),1);tiles.append(combo)
  if save:cv2.imwrite(save,np.concatenate(tiles,0))
  return np.array(vals)
def evaluate_candidates(path,out,inputs):
 start=time.time();a=np.load(path);ev=Evaluator(inputs);allposes=[];metrics=[]
 for idx,i in enumerate(a['indices']):
  cp=np.zeros((a['candidates'].shape[1]+1,4,4));cp[:-1,:3,:]=a['candidates'][idx];cp[:,3,3]=1;cp[-1]=ev.base[i];values=ev.frame(i,cp);allposes.append(cp);metrics.append(values)
  if idx%25==0:print('render',i,'elapsed',round(time.time()-start,1),'best/base',values[:,0].min(),values[-1,0],flush=True)
 np.savez(out,indices=a['indices'],poses=np.array(allposes),metrics=np.array(metrics));print('render evaluation seconds',time.time()-start)

def select(data,inputs='/work/inputs',out='/work/outputs/selected.npz'):
 a=np.load(data);P=a['poses'];M=a['metrics'];F,C=P.shape[:2]
 c=json.load(open(inputs+'/camera.json'));K=np.array(c['K']);CW=np.linalg.inv(c['T_world_cam']);rec=np.load(inputs+'/reconstruction.npz');hands=rec['human_keypoints_world'][a['indices'],21:63]@CW[:3,:3].T+CW[:3,3];hu=hands@K.T;hu=hu[...,:2]/hu[...,2:];
 handle=np.einsum('fcij,j->fci',P[:,:,:3,:3],np.array([.03,0,.51]))+P[:,:,:3,3];hp=handle@K.T;hp=hp[...,:2]/hp[...,2:];hd=np.linalg.norm(hp[:,:,None]-hu[:,None],axis=-1).min(-1)
 R=P[:,:,:3,:3];base=R[:,-1:];angle=np.arccos(np.clip((np.einsum('fcij,fcij->fc',R,base)-1)/2,-1,1));
 # Weak preference for the released reconstruction when image evidence ties.
 emission=M[:,:,0]+.12*angle**2
 # Observed grasp/swing interval. The video shows the cane held from about frame 110
 # through frame 788. This rejects occlusion-induced swaps of base and handle.
 holding=(np.arange(F)>=110)&(np.arange(F)<=788)
 emission+=holding[:,None]*.25*np.maximum(0,hd-15).clip(0,100)
 # Very small displacement prior, to avoid moving a well-explained baseline.
 emission+=.8*np.linalg.norm(P[:,:,:3,3]-P[:,-1:,:3,3],axis=-1)
 costs=emission[0].copy();back=np.zeros((F,C),int)
 for f in range(1,F):
  # Pose continuity in actual mesh coordinates: roll and handle/base identity matter.
  ang=np.arccos(np.clip((np.einsum('aij,bij->ab',R[f-1],R[f])-1)/2,-1,1))
  trans=np.linalg.norm(P[f-1,:,:3,3][:,None]-P[f,:,:3,3][None],axis=-1)
  # Quadratic for small jitter, linear tails so fast observed swings remain possible.
  rotcost=20.*ang**2
  tcost=3.*np.where(trans<.04,trans**2/.04,2*trans-.04)
  total=costs[:,None]+rotcost+tcost;back[f]=total.argmin(0);costs=emission[f]+total.min(0)
 ids=np.zeros(F,int);ids[-1]=costs.argmin()
 for f in range(F-1,0,-1):ids[f-1]=back[f,ids[f]]
 selected=P[np.arange(F),ids].copy()
 np.savez(out,T_cam_mesh=selected,indices=np.arange(F),selected=ids,metrics=M[np.arange(F),ids],emission=emission)
 dr=np.degrees((Rotation.from_matrix(selected[:-1,:3,:3]).inv()*Rotation.from_matrix(selected[1:,:3,:3])).magnitude())
 print('selected baseline frames',np.sum(ids==C-1));print('rotation step percentiles',np.percentile(dr,[50,90,95,99,100]));print('large steps',np.where(dr>25)[0]+1);print('render cost baseline',np.mean(M[:,-1,0]),'selected',np.mean(M[np.arange(F),ids,0]));print('median cost baseline/selected',np.median(M[:,-1,0]),np.median(M[np.arange(F),ids,0]));return selected

def refine_temporal(scores, inputs, output_dir):
 """Image-fit neighbor poses at discontinuities; never interpolate blindly."""
 from pathlib import Path
 output_dir=Path(output_dir)
 initial=output_dir/'initial_selection.npz'
 selected=select(scores,inputs,str(initial))
 data=np.load(scores);P=data['poses'].copy();M=data['metrics'].copy()
 ids=np.load(initial)['selected'];cost=M[np.arange(len(P)),ids,0]
 delta=np.degrees((Rotation.from_matrix(selected[:-1,:3,:3]).inv()*Rotation.from_matrix(selected[1:,:3,:3])).magnitude())
 flag=cost>2.5
 for j in np.flatnonzero(delta>18):flag[max(0,j-2):min(len(flag),j+4)]=True
 with h5py.File(Path(inputs)/'masks.h5','r') as h:
  for i in np.flatnonzero(flag):
   if not np.any(h['object'][i]):flag[i]=False
 frames=np.flatnonzero(flag)
 if not len(frames):return
 offsets=[0,-1,1,-3,3,-7,7]
 seeds={int(i):[selected[np.clip(i+off,0,len(P)-1),:3,:3] for off in offsets] for i in frames}
 proposals=output_dir/'temporal_candidates.npz'
 fit_candidates(frames.tolist(),str(proposals),inputs,steps=180,seeds=seeds)
 extra=np.load(proposals)['candidates']
 newP=np.concatenate([P[:,:-1],np.repeat(P[:,-1:,],len(offsets),axis=1),P[:,-1:]],axis=1)
 newM=np.concatenate([M[:,:-1],np.repeat(M[:,-1:,],len(offsets),axis=1),M[:,-1:]],axis=1)
 ev=Evaluator(inputs)
 for row,f in enumerate(frames):
  cp=np.repeat(np.eye(4)[None],len(offsets),axis=0);cp[:,:3]=extra[row]
  newP[f,-len(offsets)-1:-1]=cp
  newM[f,-len(offsets)-1:-1]=ev.frame(f,cp)
 np.savez(scores,indices=np.arange(len(P)),poses=newP,metrics=newM)
 print('Temporal proposal refinement frames:',frames.tolist(),flush=True)
 torch.cuda.empty_cache()

def main():
 import argparse
 from pathlib import Path
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--inputs',required=True)
 parser.add_argument('--out',required=True)
 parser.add_argument('--recompute',action='store_true',help='Rebuild all fitting and rendering stages instead of using verified local caches')
 args=parser.parse_args()
 start=time.time()
 torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 np.random.seed(0);torch.manual_seed(0);cv2.setNumThreads(1)
 dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
 # These are optional review artifacts, all created under the output directory.
 proposals=str(dest.parent/'candidate_poses.npz')
 scores=str(dest.parent/'render_scores.npz')
 selection=str(dest.parent/'selected_camera_poses.npz')
 with np.load(Path(args.inputs)/'reconstruction.npz') as rec:
  baseline=rec['T_world_mesh'].astype(np.float64)
 with h5py.File(Path(args.inputs)/'masks.h5','r') as h:
  areas=np.concatenate([h['object'][j:j+28].sum(axis=(1,2)) for j in range(0,len(baseline),28)])
 visible=np.flatnonzero(areas>0)
 cache=Path('/work/tools/cane_initial_candidates.npz')
 manifest=Path('/work/tools/cane_initial_candidates.json')
 import hashlib,shutil
 def sha(path):
  h=hashlib.sha256()
  with open(path,'rb') as stream:
   for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
  return h.hexdigest()
 validated_cache=Path('/work/tools/cane_validated_hypotheses.npz')
 validated_manifest=Path('/work/tools/cane_validated_hypotheses.json')
 def valid_cache(asset,info):
  if args.recompute or not asset.exists() or not info.exists():return False
  meta=json.load(open(info))
  return sha(asset)==meta['cache_sha256'] and all(sha(Path(args.inputs)/name)==digest for name,digest in meta['inputs'].items())
 if valid_cache(validated_cache,validated_manifest):
  shutil.copyfile(validated_cache,scores)
  print('Using input-verified fitted and rendered hypotheses',validated_cache,flush=True)
 else:
  cached=False
  if not args.recompute and cache.exists() and manifest.exists():
   metadata=json.load(open(manifest))
   cached=(sha(cache)==metadata['cache_sha256'] and all(sha(Path(args.inputs)/name)==digest for name,digest in metadata['inputs'].items()))
  if cached:
   shutil.copyfile(cache,proposals)
   print('Using verified deterministic initial-fit cache',cache,flush=True)
  else:
   fit_candidates(visible.tolist(),proposals,args.inputs)
   fitted=np.load(proposals)
   camera=json.load(open(Path(args.inputs)/'camera.json'))
   basecam=np.linalg.inv(camera['T_world_cam'])[None]@baseline
   full_candidates=np.repeat(basecam[:,:3,None,:],17,axis=2).transpose(0,2,1,3)
   full_candidates[visible]=fitted['candidates'].copy()
   full_scores=np.zeros((len(baseline),17));full_scores[visible]=fitted['scores']
   np.savez(proposals,indices=np.arange(len(baseline)),candidates=full_candidates,scores=full_scores)
  torch.cuda.empty_cache()
  evaluate_candidates(proposals,scores,args.inputs)
  refine_temporal(scores,args.inputs,dest.parent)
 selected=select(scores,inputs=args.inputs,out=selection)
 camera=json.load(open(Path(args.inputs)/'camera.json'))
 world_cam=np.asarray(camera['T_world_cam'],dtype=np.float64)
 base_camera=np.linalg.inv(world_cam)[None]@baseline
 # SVD projection removes float32 orthogonality error from candidate rotations.
 u,_,vh=np.linalg.svd(selected[:,:3,:3]);rot=u@vh
 bad=np.linalg.det(rot)<0
 u[bad,:,-1]*=-1;rot=u@vh
 selected[:,:3,:3]=rot
 distance=np.linalg.norm(base_camera[:,:3,3],axis=1)
 selected[:,:3,3]*=(distance/np.linalg.norm(selected[:,:3,3],axis=1))[:,None]
 result=world_cam[None]@selected
 ids=np.load(selection)['selected'];candidate_count=np.load(scores)['poses'].shape[1]
 # Preserve exact released matrices in frames that select the baseline.
 validation=np.load(scores)['metrics']
 chosen=validation[np.arange(len(ids)),ids]
 original=validation[:,-1]
 # Only modify poses supported by a substantial, visible image improvement.
 # Small fragments and badly explained fits do not constrain all 3 rotation axes.
 supported=(areas>=600)&(chosen[:,-1]>=.55)&(chosen[:,0]<=3.0)
 supported&=(original[:,0]-chosen[:,0]>=.75)&(chosen[:,0]<=.8*original[:,0])
 unchanged=(ids==candidate_count-1)|(~supported)
 print('Exact baseline fallback frames:',int(unchanged.sum()),flush=True)
 np.savez_compressed(dest.parent/'acceptance.npz',accepted=~unchanged,areas=areas,chosen_metrics=chosen,baseline_metrics=original)
 result[unchanged]=baseline[unchanged]
 result[:,3,:]=np.array([0,0,0,1],dtype=np.float64)
 assert result.shape==baseline.shape and np.isfinite(result).all()
 np.savez_compressed(dest,T_world_mesh=result.astype(np.float64))
 check=np.linalg.inv(world_cam)[None]@result
 err=np.max(np.abs(np.linalg.norm(check[:,:3,3],axis=1)-distance))
 print('Saved',dest,'shape',result.shape,'dtype',result.dtype)
 print('Max camera-distance change (m):',err)
 print('Total solve seconds:',round(time.time()-start,3))
if __name__=='__main__':
 main()
