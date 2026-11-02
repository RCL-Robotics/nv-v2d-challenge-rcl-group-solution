import sys,time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rot
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
center=(mesh.bounds[0]+mesh.bounds[1])/2
sc=0.5; R=Renderer(mesh,D['K'],D['W'],D['H'],scale=sc)
Tc=np.linalg.inv(D['T_wc'])[None]@D['rec']['T_world_mesh'].astype(np.float64)
frames=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
om=read_masks('/work/inputs/masks.h5','object',sc,frames); pm=read_masks('/work/inputs/masks.h5','person',sc,frames)
tgt=torch.tensor(om,dtype=torch.float32,device='cuda'); valid=torch.tensor(~(pm&~om),dtype=torch.float32,device='cuda')
t0=time.time()
T,L=fit_poses(R,Tc[frames],tgt,valid,center,iters=200)
torch.cuda.synchronize(); print('time',time.time()-t0, torch.cuda.max_memory_allocated()/1e9)
for k,i in enumerate(frames):
    dR=np.degrees(np.linalg.norm(Rot.from_matrix(T[k,:3,:3]@Tc[i,:3,:3].T).as_rotvec()))
    print(i, 'dR %.2f'%dR, 'base t',np.round(Tc[i,:3,3],3),'fit t',np.round(T[k,:3,3],3),'L %.4f'%L[k])
np.save('/work/scratch/exp/test_sil.npy',T)
