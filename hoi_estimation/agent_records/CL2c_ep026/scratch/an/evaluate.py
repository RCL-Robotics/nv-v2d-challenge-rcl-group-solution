from common import *; from render import Renderer; from scorer import *; import sys, torch
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
R=Renderer(m,K); msk=h5py.File('/work/inputs/masks.h5'); S=Scorer(R,'/work/scratch/frames',msk)
n=pl[:3]/np.linalg.norm(pl[:3]); d=pl[3]/np.linalg.norm(pl[:3])
Tb=r['T_world_mesh'].astype(float)
trajs={'baseline':Tb}
for p in sys.argv[1:]: trajs[p.split('/')[-1]]=np.load(p)['T_world_mesh']
phases=[(0,281,'S1'),(281,357,'M1'),(357,399,'S2'),(399,510,'M2'),(510,669,'S3')]
F=len(Tb); v=m.vertices[::25]
stats={}
for name,T in trajs.items():
    Tc=Tcw[None]@T; sc=np.zeros((F,2))
    for i in range(F):
        sc[i]=np.array(S.score(i,Tc[i:i+1],crop_for(msk['object'][i]))).ravel()
    # wheel color test
    stats[name]=(sc,Tc)
    np.save(f'/work/scratch/an/eval_{name}.npy',sc)
Tcb=Tcw[None]@Tb
print('%-12s %-3s %6s %6s %8s %8s %8s %8s %7s'%('traj','ph','IoU','NCC','dz_mean','dz_med','rotdiff','dRot/fr','wheelH'))
for name,(sc,Tc) in stats.items():
    T=Twc[None]@Tc
    for a,b,pn in phases+[(0,F,'ALL')]:
        dz=(Tc[a:b,2,3]-Tcb[a:b,2,3])*100
        rd=rotang(Tc[a:b,:3,:3],Tcb[a:b,:3,:3])
        dr=rotang(Tc[a+1:b,:3,:3],Tc[a:b-1,:3,:3])
        wh=[np.sort((v@T[i,:3,:3].T+T[i,:3,3])@n+d)[:10].mean() for i in range(a,b,5)]
        print('%-12s %-3s %6.3f %6.3f %8.2f %8.2f %8.1f %8.2f %7.3f'%(name[:12],pn,sc[a:b,0].mean(),sc[a:b,1].mean(),dz.mean(),np.median(dz),np.median(rd),dr.mean(),np.mean(wh)))
