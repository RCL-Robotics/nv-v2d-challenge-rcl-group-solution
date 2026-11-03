import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
import solve
from hoilib import *
K,Twc,rec,mesh,plane=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tb=np.einsum('ij,fjk->fik',Tcw,rec['T_world_mesh'].astype(float))
d=np.load('masks_half.npz'); idx=list(range(540,604,2))
om=(d['om'][idx]/252.).astype(np.float32); pm=d['pm'][idx]>0
rd=Renderer(mesh,K,0.5)
ims=read_video('/work/inputs/video.mp4',idx,0.5)
valid=(1-(pm&(om<0.5))).astype(np.float32); edt=solve.edge_dt_maps(ims,pm&(om<0.5))
P=np.asarray(mesh.vertices)[::25]
res={}
for it,lr in [(200,2e-3),(400,3e-3),(800,3e-3)]:
    T,i0,i1=solve.fit_frames(rd,Tb[idx],om,valid,0,edt=edt,w_edge=0.3,iters=it,lr=lr)
    res.setdefault((it,lr),[]).append(T); print(it,lr,'IoU',i1.mean())
np.save('conv_res.npy',np.stack([v[0] for v in res.values()]))
k=list(res)
def norm(T): return rescale_depth(T, Tb[idx,2,3])
for k,v in res.items():
    print(k,'zratio', np.median(v[0][:,2,3]/Tb[idx,2,3]).round(3), 'rot vs base', np.median(rotdeg(v[0][:,:3,:3],Tb[idx,:3,:3])).round(2))
c=res[k[2]][0]
for a,b,l in [(res[k[0]][0],c,'200 vs 800'),(res[k[1]][0],c,'400 vs 800')]:
  print(l,'rot med %.2f, ADD after depth-normalising %.2f cm'%(np.median(rotdeg(a[:,:3,:3],b[:,:3,:3])),(add_err(norm(a),norm(b),P)*100).mean()))

