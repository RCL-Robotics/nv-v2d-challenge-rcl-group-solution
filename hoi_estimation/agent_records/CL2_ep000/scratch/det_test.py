import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
import solve
from hoilib import *
K,Twc,rec,mesh,plane=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tb=np.einsum('ij,fjk->fik',Tcw,rec['T_world_mesh'].astype(float))
d=np.load('masks_half.npz'); idx=list(range(400,432))
om=(d['om'][idx]/252.).astype(np.float32); pm=d['pm'][idx]>0
rd=Renderer(mesh,K,0.5)
ims=read_video('/work/inputs/video.mp4',idx,0.5)
valid=(1-(pm&(om<0.5))).astype(np.float32); edt=solve.edge_dt_maps(ims,pm&(om<0.5))
R=[solve.fit_frames(rd,Tb[idx],om,valid,0,edt=edt,w_edge=0.3)[0] for _ in range(2)]
print('max rot diff run1-run2 %.4f deg, max trans diff %.3f mm'%(rotdeg(R[0][:,:3,:3],R[1][:,:3,:3]).max(), np.abs(R[0][:,:3,3]-R[1][:,:3,3]).max()*1000))
S=[solve.fit_frames(rd,Tb[idx][::2],om[::2],valid[::2],0,shared=True,edt=edt[::2],w_edge=0.3,ks=(5,3,1))[0] for _ in range(2)]
print('shared: rot diff %.4f deg trans %.3f mm'%(rotdeg(S[0][0,:3,:3],S[1][0,:3,:3]), np.abs(S[0][0,:3,3]-S[1][0,:3,3]).max()*1000))
