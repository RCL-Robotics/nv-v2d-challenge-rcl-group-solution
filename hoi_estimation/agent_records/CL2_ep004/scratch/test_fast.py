import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2, time
from bat_lib import *
mesh=load_mesh('/work/inputs'); t0=time.time()
vs,fs,cs=simplified_mesh(mesh); print(vs.shape,fs.shape,time.time()-t0)
np.savez('/work/scratch/simp.npz',vs=vs,fs=fs,cs=cs)
FR=FastRenderer(vs,fs,cs)
K,Twc,rec,plane=load_inputs('/work/inputs')
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'][0].astype(np.float64)
# crop around projection
p=K@Tc[:3,3]; p=p[:2]/p[2]; side=400; x0,y0=p[0]-side/2,p[1]-side/2; res=256
Kc=K.copy(); Kc[0,2]-=x0; Kc[1,2]-=y0; Kc[:2]*=res/side
R=torch.tensor(Tc[None,:3,:3],dtype=torch.float32,device='cuda'); t=torch.tensor(Tc[None,:3,3],dtype=torch.float32,device='cuda')
m,c,d=FR.render(R,t,Kc[None],res)
cv2.imwrite('/work/scratch/fast.png',np.hstack([(m[0].cpu().numpy()[...,None]*255).repeat(3,2),(c[0].cpu().numpy()[...,::-1]*255)]).astype(np.uint8))
torch.cuda.synchronize(); t0=time.time()
Rb=R.repeat(32,1,1); tb=t.repeat(32,1); Kb=np.repeat(Kc[None],32,0)
for _ in range(20):
    Rb.requires_grad_(True); m,c,d=FR.render(Rb,tb,Kb,res); (m.sum()+c.sum()).backward()
torch.cuda.synchronize(); print('per iter B=32', (time.time()-t0)/20)
