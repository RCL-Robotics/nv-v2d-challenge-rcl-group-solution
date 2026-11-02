# per-frame silhouette fit from baseline init; saves scratch/silfit.npz
from rend import *
import time
r,cam=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)[None]@r['T_world_mesh'].astype(float)
F=len(Tc); M=Mesh(); MK=Masks(); S=192
ms=np.load('maskstats.npz'); area=ms['area']
p=Tc[:,:3,3]; uvb=(K@p.T).T; uvb=uvb[:,:2]/uvb[:,2:]
# person mask dilation to ignore boundary pixels near occluders
ker=np.ones((5,5),np.uint8)
out={k:[] for k in ['R','t','iou','iou0','x0y0']}
t0=time.time(); B=48
for s in range(0,F,B):
    ids=list(range(s,min(F,s+B)))
    x0y0=np.array([[int(np.clip(uvb[i,0]-S/2,0,1536-S)),int(np.clip(uvb[i,1]-S/2,0,1152-S))] for i in ids])
    O=[];V=[]
    for k,i in enumerate(ids):
        o,pm=MK.crop(i,*x0y0[k],S)
        pmd=cv2.dilate(pm.astype(np.uint8),ker)>0
        O.append(o); V.append(~pmd|o)
    O=torch.tensor(np.array(O),dtype=torch.float32,device=DEV); V=torch.tensor(np.array(V),dtype=torch.float32,device=DEV)
    R,t,iou,_=fit_sil(M,K,Tc[ids,:3,:3],uvb[ids],p[ids,2],O,V,x0y0,S)
    _,_,iou0,_=fit_sil(M,K,Tc[ids,:3,:3],uvb[ids],p[ids,2],O,V,x0y0,S,iters=(0,0,0))
    out['R'].append(R); out['t'].append(t); out['iou'].append(iou); out['iou0'].append(iou0); out['x0y0'].append(x0y0)
    print(s, time.time()-t0, iou0.mean(), iou.mean(), flush=True)
np.savez('silfit.npz',**{k:np.concatenate(v) for k,v in out.items()})
