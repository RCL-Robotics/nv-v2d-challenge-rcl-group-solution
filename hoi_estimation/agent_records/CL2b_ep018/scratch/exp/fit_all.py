import sys,time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoi_lib import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
center=(mesh.bounds[0]+mesh.bounds[1])/2
sc=0.5; R=Renderer(mesh,D['K'],D['W'],D['H'],scale=sc)
Tc=np.linalg.inv(D['T_wc'])[None]@D['rec']['T_world_mesh'].astype(np.float64)
OM=np.load('om_half.npy'); PM=np.load('pm_half.npy'); BB=np.load('bb.npy')
out=np.zeros((858,4,4)); Ls=np.zeros(858)
t0=time.time()
for s in range(0,858,48):
    frames=list(range(s,min(s+48,858)))
    om=OM[frames]; pm=PM[frames]; bb=BB[frames]
    cen=np.stack([(bb[:,0]+bb[:,1])/2,(bb[:,2]+bb[:,3])/2],1)
    off,(omc,pmc)=make_crops([om,pm],cen)
    tgt=torch.tensor(omc,dtype=torch.float32,device='cuda'); valid=torch.tensor(~(pmc&~omc),dtype=torch.float32,device='cuda')
    T,L=fit_poses(R,Tc[frames],tgt,valid,center,iters=200,crop=(torch.tensor(off,device='cuda',dtype=torch.float32),(192,256)))
    out[frames]=T; Ls[frames]=L
torch.cuda.synchronize(); print('time',time.time()-t0)
np.savez('fit_all.npz',T=out,L=Ls)
