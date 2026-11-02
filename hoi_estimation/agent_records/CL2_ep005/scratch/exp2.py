import sys, time, os; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, h5py, trimesh
from hoi_lib import *
from traj_opt import *
from scipy.spatial.transform import Rotation as Rot
torch.manual_seed(0)
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh); SM=SilMesh(mesh,glctx=R.glctx)
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T); F=len(Tc)
z=np.load('/work/scratch/crops.npz'); crops,om,vm=z['crops'],z['om'],z['vm']; S=192
static=[(0,158),(583,713),(938,1002)]
node_of=np.zeros(F,int); init=[]; n=0; i=0
while i<F:
    g=[s for s in static if s[0]==i]
    if g:
        a,b=g[0]; Rm=Rot.from_matrix(Tc[a:b+1,:3,:3]).mean().as_matrix(); tt=Tc[a:b+1,:3,3]
        xy=np.median(tt[:,:2]/tt[:,2:],0); zz=np.median(tt[:,2])
        Ti=np.eye(4); Ti[:3,:3]=Rm; Ti[:3,3]=[xy[0]*zz,xy[1]*zz,zz]; init.append(Ti); node_of[a:b+1]=n; n+=1; i=b+1
    else:
        init.append(Tc[i]); node_of[i]=n; n+=1; i+=1
init=np.stack(init)
lam=10.0
def smooth(Rn,t):
    aR=Rn[:-2]-2*Rn[1:-1]+Rn[2:]; at=t[:-2]-2*t[1:-1]+t[2:]
    return lam*((aR**2).sum()+(at**2).sum())
fw=np.ones(F)/np.bincount(node_of)[node_of]**0.5
opt=TrajOpt(SM,d['K'],init,node_of,crops,om,vm,S)
if os.path.exists('/work/scratch/stage1.npz'):
    st=np.load('/work/scratch/stage1.npz'); [p.data.copy_(torch.tensor(st[k],device='cuda')) for k,p in zip(['w','dxy','dlz'],[opt.w,opt.dxy,opt.dlz])]
else:
    t=time.time(); opt.run(150,fw,smooth,log=print); print('stage1',time.time()-t)
    np.savez('/work/scratch/stage1.npz',w=opt.w.detach().cpu().numpy(),dxy=opt.dxy.detach().cpu().numpy(),dlz=opt.dlz.detach().cpu().numpy())
    np.save('/work/scratch/stage1_T.npy',opt.frame_T().detach().cpu().numpy().astype(np.float64))
E=320
if os.path.exists('/work/scratch/edges.npz'):
    z=np.load('/work/scratch/edges.npz'); dts,pms=z['dts'],z['pms']
else:
    G=np.load('/work/scratch/frames_full_gray.npy',mmap_mode='r'); PM=h5py.File('/work/inputs/masks.h5')['person']
    t=time.time(); dts,pms=make_edge_maps(crops,lambda i:np.ascontiguousarray(G[i]),lambda i:PM[i],E=E); print('edges',time.time()-t)
    np.savez('/work/scratch/edges.npz',dts=dts,pms=pms)
Sobb,_=trimesh.bounds.oriented_bounds(mesh); Robb=np.linalg.inv(Sobb)[:3,:3]
v=SM.v.cpu().numpy(); f_=SM.f.cpu().numpy().astype(int); fnm=np.cross(v[f_[:,1]]-v[f_[:,0]],v[f_[:,2]]-v[f_[:,0]]); fnm/=np.linalg.norm(fnm,axis=1,keepdims=True)+1e-12
fn_obb=torch.tensor(np.argmax(fnm@np.concatenate([Robb,-Robb],1),1)+1,device='cuda')
mu=float(sys.argv[1]); ew=np.zeros(F); ew[0::2]=1.0
ew=ew/np.bincount(node_of)[node_of]**0.5
edge=dict(dts=torch.tensor(dts,device='cuda'),pms=torch.tensor(pms,device='cuda'),E=E,fn_obb=fn_obb,mu=mu)
t=time.time(); opt.run(100,fw,smooth,lr=1e-3,log=print,edge=edge,edge_w=ew); print('stage2',time.time()-t,'mem',torch.cuda.max_memory_allocated()/1e9)
np.save(f'/work/scratch/stage2_{mu}.npy',opt.frame_T().detach().cpu().numpy().astype(np.float64))
print('iou',opt.ious().mean())
