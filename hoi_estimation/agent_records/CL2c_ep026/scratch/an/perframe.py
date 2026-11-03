from common import *; from render import Renderer; from scorer import *; from param import *; import time, torch, sys
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
R=Renderer(m,K); msk=h5py.File('/work/inputs/masks.h5'); S=Scorer(R,'/work/scratch/frames',msk)
fp=FloorParam(pl[:3])
Tb=r['T_world_mesh'].astype(float); Ry=np.diag([-1,1,-1,1.])
def search(i,T0,crop):
    best=(0,0,0)
    for ys,ab in [(np.radians(np.arange(-12,13,3)),np.arange(-0.045,0.046,0.015)),(np.radians(np.arange(-2,2.1,1)),np.arange(-0.01,0.011,0.005)),(np.radians(np.arange(-0.5,0.6,0.5)),np.arange(-0.0025,0.0026,0.0025))]:
        C=[(best[0]+y,best[1]+a,best[2]+b) for y in ys for a in ab for b in ab]
        P=np.stack([Tcw@fp.apply(T0,*c) for c in C])
        iou,ncc=S.score(i,P,crop); s=iou+ncc; k=s.argmax(); best=C[k]
    return best, iou[k], ncc[k]
F=len(Tb); out=np.full((F,2,5),np.nan)
t=time.time()
for i in range(F):
    ob=msk['object'][i]
    if ob.sum()<100: continue
    crop=crop_for(ob)
    for h,T0 in enumerate([Tb[i]@Ry,Tb[i]]):
        if h==1 and i%6: continue
        b,iou,ncc=search(i,T0,crop); out[i,h]=[*b,iou,ncc]
    if i%50==0: print(i,time.time()-t,out[i].round(3),flush=True)
np.save('/work/scratch/an/perframe.npy',out)
