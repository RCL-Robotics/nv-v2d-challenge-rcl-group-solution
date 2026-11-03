from common import *; from render import Renderer; import cv2, torch
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
R=Renderer(m,K)
T=r['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@T
Ry=np.diag([-1,1,-1,1.]).astype(float)
msk=h5py.File('/work/inputs/masks.h5')
F=len(T); out=np.zeros((F,2)); iou=np.zeros((F,2))
def ncc(a,b,w):
    w=w.astype(bool)
    if w.sum()<30: return np.nan
    a=a[w]-a[w].mean(); b=b[w]-b[w].mean()
    return (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9)
for i in range(F):
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[...,::-1].astype(np.float32)/255
    per=msk['person'][i]; ob=msk['object'][i]
    cols,masks,_=R.render(np.stack([Tc[i],Tc[i]@Ry]))
    g=im.mean(-1)
    for k in range(2):
        c=cols[k].cpu().numpy(); mk=masks[k].cpu().numpy()>0
        out[i,k]=ncc(c.mean(-1),g,mk&~per)
        iou[i,k]=(mk&ob).sum()/max((mk|ob).sum(),1)
np.savez('/work/scratch/an/flipscore.npz',ncc=out,iou=iou)
for a in range(0,F,30):
    b=min(a+30,F); print(a,b,np.nanmean(out[a:b],0).round(3), np.nanmean(out[a:b,0]>out[a:b,1]).round(2), iou[a:b].mean(0).round(3))
