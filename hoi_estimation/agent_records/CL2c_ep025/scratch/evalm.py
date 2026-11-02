import numpy as np, h5py, torch, json
from rutil import *
R=Renderer('/work/inputs/object_mesh.glb')
K,Twc,Kp=load_cam(); Tcw=np.linalg.inv(Twc)
mf=h5py.File('/work/inputs/masks.h5')
OM=mf['object'][:]; PM=mf['person'][:]
F=len(OM)
cent=np.zeros((F,2)); area=OM.reshape(F,-1).sum(1)
for i in range(F):
    ys,xs=np.nonzero(OM[i])
    cent[i]=(xs.mean(),ys.mean()) if len(xs) else (np.nan,np.nan)
S=256
def iou_all(Tc, KK=K):
    out=np.zeros(F)
    for i in range(F):
        p=KK@Tc[i,:3,3]; c=cent[i] if area[i]>0 else p[:2]/p[2]
        x0=int(np.clip(c[0]-S/2,0,1536-S)); y0=int(np.clip(c[1]-S/2,0,1152-S))
        _,m,_=R.render(Tc[i],KK,S,S,x0,y0); m=m[0].cpu().numpy()
        o=OM[i,y0:y0+S,x0:x0+S]; pm=PM[i,y0:y0+S,x0:x0+S]&~o
        valid=~pm
        inter=(m&o).sum(); uni=((m|o)&valid).sum()
        out[i]=inter/max(uni,1)
    return out
if __name__=='__main__':
    r=np.load('/work/inputs/reconstruction.npz')
    Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
    ib=iou_all(Tb); ic=iou_all(r['T_cam_mesh_coconet'].astype(np.float64),Kp); iff=iou_all(r['T_cam_mesh_foundationpose'].astype(np.float64),Kp)
    np.savez('iou_base.npz',b=ib,c=ic,f=iff,area=area)
    for s in range(0,F,30):
        e=slice(s,s+30); print(s, 'area %5d'%area[e].mean(), 'base %.3f coco %.3f fp %.3f'%(ib[e].mean(),ic[e].mean(),iff[e].mean()))
    print('mean',ib.mean(),ic.mean(),iff.mean())
