import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
import solve
from hoilib import *
K,Twc,rec,mesh,plane=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tb=np.einsum('ij,fjk->fik',Tcw,rec['T_world_mesh'].astype(float))
Tf=np.load('cache/pf.npz')['T']
d=np.load('masks_half.npz')
rd=Renderer(mesh,K,0.5)
hv=rec['human_vertices_world'].astype(float); hum=np.einsum('ij,fnj->fni',Tcw[:3,:3],hv)+Tcw[:3,3]
for name,ph in [('A',np.arange(0,275)),('B',np.arange(622,778))]:
    sel=np.unique(np.linspace(ph[0],ph[-1],48).astype(int))
    om=(d['om'][sel]/252.).astype(np.float32); pm=d['pm'][sel]>0
    occ=solve.person_in_front(rd,Tb[sel],hum[sel],pm)&(om<0.5)
    valid=(1-occ).astype(np.float32)
    edt=solve.edge_dt_maps(read_video('/work/inputs/video.mp4',list(sel),0.5),occ)
    init=robust_mean_pose(Tf[ph])
    for iters,lr in [(200,2e-3),(200,2e-3),(400,1e-3),(400,1e-3)]:
        T=solve.fit_frames(rd,np.repeat(init[None],len(sel),0),om,valid,0,shared=True,edt=edt,w_edge=0.3,ks=(5,3,1),iters=iters,lr=lr)[0][0]
        print(name,iters,lr,'vs init %.2f deg, dz %.1f cm'%(rotdeg(T[:3,:3],init[:3,:3]),(T[2,3]-init[2,3])*100), 'R row', np.round(Rot.from_matrix(T[:3,:3]@init[:3,:3].T).as_rotvec()*57.3,2))
