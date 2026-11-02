import sys, time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, h5py
from hoi_lib import *
from traj_opt import *
from scipy.spatial.transform import Rotation as Rot
torch.manual_seed(0)
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T); F=len(Tc)
f=h5py.File('/work/inputs/masks.h5')
S=192
import os
if os.path.exists('crops.npz'):
    z=np.load('crops.npz'); crops,om,vm=z['crops'],z['om'],z['vm']
else:
    t=time.time(); OM=f['object'][:]; PM=f['person'][:]
    crops,om,vm=make_crops(Tc,lambda i:OM[i],lambda i:PM[i],d['K'],d['H'],d['W'],R,S=S)
    np.savez('crops.npz',crops=crops,om=om,vm=vm); print('crops',time.time()-t)
static=[(0,158),(583,713),(938,1002)]
node_of=np.zeros(F,int); init=[]; n=0; i=0
while i<F:
    g=[s for s in static if s[0]==i]
    if g:
        a,b=g[0]; Rm=Rot.from_matrix(Tc[a:b+1,:3,:3]).mean().as_matrix(); tt=Tc[a:b+1,:3,3]
        xy=np.median(tt[:,:2]/tt[:,2:],0); z=np.median(tt[:,2])
        Ti=np.eye(4); Ti[:3,:3]=Rm; Ti[:3,3]=[xy[0]*z,xy[1]*z,z]; init.append(Ti); node_of[a:b+1]=n; n+=1; i=b+1
    else:
        init.append(Tc[i]); node_of[i]=n; n+=1; i+=1
init=np.stack(init)
lam=float(sys.argv[1])
def smooth(Rn,t):
    aR=Rn[:-2]-2*Rn[1:-1]+Rn[2:]; at=t[:-2]-2*t[1:-1]+t[2:]
    return lam*((aR**2).sum()+(at**2).sum())
fw=np.ones(F)
fw=fw/np.bincount(node_of)[node_of]**0.5  # static groups: total weight ~ sqrt(n) frames
ho=np.zeros(F,bool)
if '--holdout' in sys.argv:
    mv=np.where(np.bincount(node_of)[node_of]==1)[0]; ho[mv[2::5]]=True; fw[ho]=0

SM=SilMesh(mesh,glctx=R.glctx)
opt=TrajOpt(SM,d['K'],init,node_of,crops,om,vm,S)
i0=opt.ious()
t=time.time(); opt.run(200,fw,smooth,log=print); print('opt time',time.time()-t,'mem',torch.cuda.max_memory_allocated()/1e9)
i1=opt.ious()
Tout=opt.frame_T().detach().cpu().numpy().astype(np.float64)
np.save(f'exp1_{lam}'+('_ho' if '--holdout' in sys.argv else '')+'.npy',Tout)
print('iou init',i0.mean(),'final',i1.mean(),'holdout init',i0[ho].mean(),'final',i1[ho].mean())
for a in range(0,F,100): print(a, i0[a:a+100].mean().round(3), i1[a:a+100].mean().round(3), 'rot chg',np.median(rot_angle(Tc[a:a+100,:3,:3],Tout[a:a+100,:3,:3])).round(2),'z ratio',np.median(Tout[a:a+100,2,3]/Tc[a:a+100,2,3]).round(4))
