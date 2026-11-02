from common import *; from render import Renderer; from scorer import *; from param import *; import time, torch
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
R=Renderer(m,K); msk=h5py.File('/work/inputs/masks.h5'); S=Scorer(R,'/work/scratch/frames',msk)
T=r['T_world_mesh'].astype(float)@np.diag([-1,1,-1,1.]); fp=FloorParam(pl[:3])
for i in [100,600,330]:
    crop=crop_for(msk['object'][i])
    t=time.time()
    ys=np.radians(np.arange(-12,13,3)); ab=np.arange(-0.045,0.046,0.015)
    C=[(y,a,b) for y in ys for a in ab for b in ab]
    P=np.stack([Tcw@fp.apply(T[i],*c) for c in C])
    iou,ncc=S.score(i,P,crop); torch.cuda.synchronize()
    s=iou+ncc; k=s.argmax()
    print(i,crop,len(C),'%.2fs'%(time.time()-t), 'best',np.degrees(C[k][0]),C[k][1:],iou[k].round(3),ncc[k].round(3),' center',iou[len(C)//2].round(3),ncc[len(C)//2].round(3), torch.cuda.max_memory_allocated()/1e9)
