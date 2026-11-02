import sys; sys.path.insert(0,'/work/scratch'); sys.path.insert(0,'/work/outputs')
from static_test import *
cxy=np.array([0.005,-0.004,0])
Fz=np.eye(4); Fz[:3,:3]=np.diag([-1,-1,1.]); Fz[:3,3]=2*cxy  # p -> Rz180 (p - c) + c
idx=list(range(0,715,12))
Tb=Tc0.copy(); Tf=Tc0@Fz
res={}
for name,Tc in [('base',Tb),('flip',Tf)]:
    _,iou0,ls0,lp0=refine_fz(idx,Tc,iters=1,lam=0.3)
    Tn,iou,ls,lp=refine_fz(idx,Tc,iters=100,lam=0.0)
    # evaluate photo at refined
    Tfull=Tc.copy(); Tfull[idx]=Tn
    _,iou2,ls2,lp2=refine_fz(idx,Tfull,iters=1,lam=0.3)
    res[name]=(iou0,lp0,iou2,lp2)
for k,i in enumerate(idx):
    b=res['base']; f_=res['flip']
    print(i,'raw: iou %.3f/%.3f photo %.3f/%.3f | refined: iou %.3f/%.3f photo %.3f/%.3f'%(b[0][k],f_[0][k],b[1][k],f_[1][k],b[2][k],f_[2][k],b[3][k],f_[3][k]))
