import os,time,json,argparse
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,cv2,h5py,trimesh,torch
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
def prepare_mesh(path,n=1000):
 m=trimesh.load(path,force='scene',process=False).to_geometry();p,f=trimesh.sample.sample_surface(m,18000,seed=0);key=np.floor(p/.009).astype(int);_,ix=np.unique(key,axis=0,return_index=True);ix=np.sort(ix);rng=np.random.default_rng(0);ix=rng.choice(ix,min(n,len(ix)),replace=False);p=p[ix];f=f[ix]
 return m,p
def ortho6(x):
 a=torch.nn.functional.normalize(x[...,:3],dim=-1);b=x[...,3:];b=torch.nn.functional.normalize(b-(a*b).sum(-1,keepdim=True)*a,dim=-1);c=torch.cross(a,b,dim=-1);return torch.stack([a,b,c],dim=-1)
def project(R,uv,d,pts,K):
 ray=torch.cat([(uv-K[None,:2,2])/torch.tensor([K[0,0],K[1,1]],device=uv.device),torch.ones_like(uv[...,:1])],-1);t=torch.nn.functional.normalize(ray,dim=-1)*d[...,None];q=torch.einsum('bij,nj->bni',R,pts)+t[:,None];return q[...,:2]/q[...,2:]*torch.tensor([K[0,0],K[1,1]],device=uv.device)+K[None,:2,2],t
def run(indices,out,steps=160,n=800):
 start=time.time();p='/work/inputs/';a=np.load(p+'reconstruction.npz');c=json.load(open(p+'camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);base=np.linalg.inv(W)[None]@a['T_world_mesh'];m,pts=prepare_mesh(p+'object_mesh.glb',n);h=h5py.File(p+'masks.h5');device='cuda';
 allres=[];allstats=[]
 for st in range(0,len(indices),8):
  ids=indices[st:st+8];Rs=[];uvs=[];ds=[];obs=[];maps=[];scales=[];meta=[]
  for i in ids:
   mask=np.asarray(h['object'][i],np.uint8);person=np.asarray(h['person'][i],np.uint8);yx=np.argwhere(mask);xy=yx[:,::-1].astype(float);cov=np.cov(xy.T);ev,evec=np.linalg.eigh(cov);axis=evec[:,-1];
   # Rotate candidate long-axis image direction to observed mask PCA, with both signs.
   starts=[]
   for source in [base[i,:3,:3],a['T_cam_mesh_foundationpose'][i,:3,:3]]:
    for sign in [1,-1]:
     v=source[:,2]; ang=np.arctan2(sign*axis[1],sign*axis[0])-np.arctan2(v[1],v[0]);rz=Rotation.from_rotvec([0,0,ang]).as_matrix();rr=rz@source
     for zroll in [0,np.pi/2,np.pi,3*np.pi/2]:starts.append(rr@Rotation.from_rotvec([0,0,zroll]).as_matrix())
   starts=[base[i,:3,:3]]+starts
   N=len(starts);yy,xx=np.where(mask);left=max(0,min(xx.min(),int(K[0,0]*base[i,0,3]/base[i,2,3]+K[0,2]))-190);top=max(0,min(yy.min(),int(K[1,1]*base[i,1,3]/base[i,2,3]+K[1,2]))-190);right=min(1536,max(xx.max(),int(K[0,0]*base[i,0,3]/base[i,2,3]+K[0,2]))+191);bottom=min(1152,max(yy.max(),int(K[1,1]*base[i,1,3]/base[i,2,3]+K[1,2]))+191)
   # Maps use full-frame quarter resolution; distance measured in original pixels.
   merged=(mask|person);dt=cv2.distanceTransform(1-merged,cv2.DIST_L2,5);dt=cv2.resize(dt,(384,288),interpolation=cv2.INTER_AREA);pm=cv2.resize(person.astype(float),(384,288),interpolation=cv2.INTER_AREA)
   rng=np.random.default_rng(i);ix=rng.choice(len(xy),256,replace=len(xy)<256);ob=xy[ix];
   dist=np.linalg.norm(base[i,:3,3]);uv=K@base[i,:3,3];uv=uv[:2]/uv[2];
   Rs.extend(starts);uvs.extend([uv]*N);ds.extend([dist]*N);obs.extend([ob]*N);maps.extend([np.stack([dt,pm])]*N);scales.extend([np.sqrt(ev[-1])]*N);meta.append((i,N,len(xy)))
  rt=torch.tensor(np.array(Rs),dtype=torch.float32,device=device);x=torch.nn.Parameter(torch.cat([rt[:,:,0],rt[:,:,1]],dim=1));u0=torch.tensor(np.array(uvs),dtype=torch.float32,device=device);du=torch.nn.Parameter(torch.zeros_like(u0));d=torch.tensor(ds,dtype=torch.float32,device=device);ob=torch.tensor(np.array(obs),dtype=torch.float32,device=device);maps=torch.tensor(np.array(maps),dtype=torch.float32,device=device);pt=torch.tensor(pts,dtype=torch.float32,device=device);kt=torch.tensor(K,dtype=torch.float32,device=device);
  opt=torch.optim.Adam([{'params':[x],'lr':.025},{'params':[du],'lr':1.0}]);
  def loss():
   R=ortho6(x);q,t=project(R,u0+du,d,pt,kt);grid=(q/torch.tensor([1535.,1151.],device=device)*2-1)[:,None];sm=torch.nn.functional.grid_sample(maps,grid,align_corners=True,padding_mode='border')[:,:,0];dist=sm[:,0];vis=(1-sm[:,1]).clamp(0,1);fw=(torch.sqrt(dist**2+4)-2).sum(-1)/(vis.sum(-1)+80);
   D=torch.cdist(ob,q);bw=(torch.sqrt(D.min(-1).values**2+4)-2).mean(-1);
   score=.7*fw+bw;reg=.00015*(du**2).sum(-1);return score+reg,score,R,t,fw,bw
  for it in range(steps):
   opt.zero_grad();L,*_=loss();L.sum().backward();opt.step();
   if it==int(steps*.7):
    for pg in opt.param_groups:pg['lr']*=.3
  with torch.no_grad():L,s,R,t,fw,bw=loss()
  R=R.cpu().numpy();t=t.cpu().numpy();s=s.cpu().numpy();fw=fw.cpu().numpy();bw=bw.cpu().numpy();j=0
  for i,N,area in meta:
   order=np.argsort(s[j:j+N]);k=j+order[0];mat=np.eye(4);mat[:3,:3]=R[k];mat[:3,3]=t[k];allres.append(mat);allstats.append([i,s[k],fw[k],bw[k],order[0],area]);print(i,'score',np.round(s[k],2),'fw/bw',np.round([fw[k],bw[k]],2),'choice',order[:4].tolist(), 'top',np.round(s[j+order[:4]],2),'axis',np.round(R[k,:,2],2),flush=True);j+=N
  print('elapsed',round(time.time()-start,1),flush=True)
 np.savez(out,T_cam_mesh=np.array(allres),indices=indices,stats=np.array(allstats));print('saved',out,'seconds',time.time()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',default='/work/outputs/experiment.npz');p.add_argument('--steps',type=int,default=160);p.add_argument('--indices',default='0,70,140,210,245,280,315,350,385,420,455,490,525,560,595,630,665,700,735,770,805,840,875,885');args=p.parse_args();run([int(x) for x in args.indices.split(',')],args.out,args.steps)
