import sys, numpy as np, json, time
sys.path.insert(0,'/work/outputs')
from hoilib import *
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
Tb=np.linalg.inv(Twc)[None]@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
d=np.load('/work/scratch/crops.npz'); mesh=TexMesh('/work/inputs/object_mesh.glb')
FY=np.diag([-1,1,-1,1.])
torch.manual_seed(0)
t=time.time()
Ti=np.concatenate([Tb,Tb@FY]); n=len(Tb)
T,I=fit_silhouettes(mesh,np.concatenate([d['Kc']]*2),np.concatenate([d['obs']]*2),np.concatenate([d['val']]*2),Ti,iters=150,fix_z=True)
print('time',time.time()-t, torch.cuda.max_memory_allocated()/1e9)
np.savez('/work/scratch/fit_base_flip.npz',Tbase=T[:n],Tflip=T[n:],Ibase=I[:n],Iflip=I[n:])
# initial IoU
T0,I0=fit_silhouettes(mesh,d['Kc'],d['obs'],d['val'],Tb,iters=0)
for a in range(0,n,10): print(a,'iou init %.3f base %.3f flip %.3f'%(I0[a],I[a],I[n+a]), 'rot moved %.1f'%rotmat_angle(T[a][:3,:3],Tb[a][:3,:3]), 'flipback-vs-base %.1f'%rotmat_angle(T[a][:3,:3],(T[n+a]@FY)[:3,:3]))
