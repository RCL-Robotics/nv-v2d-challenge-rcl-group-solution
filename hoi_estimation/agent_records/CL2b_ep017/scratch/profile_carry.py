import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, time
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
frames=[i for i in range(110,346) if ep.obj_mask[i].sum()>300]
tg=MaskTargets(ep,frames,ep.Tc_base[frames],size=128)
grid=np.arange(0,90,5)
R0=ep.T_base[0,:3,:3]
T0=[];idx=[]
for k,i in enumerate(frames):
    Tb=ep.Tc_base[i]
    tw,_=twist_swing(R0.T@ep.T_base[i,:3,:3])
    for g in grid:
        T0.append(axis_rot(Tb,g-tw)); idx.append(k)   # absolute twist g (mod 90 grid)
T0=np.array(T0); idx=np.array(idx)
t=time.time()
Tf,iou=fit_frames(rd,ep,tg,T0,idx,iters=80,fix_twist=True,chunk=128)
print('time',time.time()-t)
np.savez('carry_profile.npz',frames=frames,grid=grid,Tf=Tf,iou=iou,area=tg.area.cpu().numpy())
