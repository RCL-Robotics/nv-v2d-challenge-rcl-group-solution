import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, time
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
frames=list(range(110,345))
tg=MaskTargets(ep,frames,ep.Tc_base[frames],size=128)
offs=[-30,0,30,60]
T0=[];idx=[]
for k,i in enumerate(frames):
    for o in offs:
        T0.append(axis_rot(ep.Tc_base[i],o)); idx.append(k)
T0=np.array(T0); idx=np.array(idx)
t=time.time()
Tf,iou=fit_frames(rd,ep,tg,T0,idx,iters=100)
print('time',time.time()-t)
np.savez('carry_fit.npz',frames=frames,offs=offs,Tf=Tf,iou=iou,idx=idx,area=tg.area.cpu().numpy())
