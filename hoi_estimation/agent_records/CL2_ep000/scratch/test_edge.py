import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch')
import numpy as np, torch, cv2, h5py
import solve
from hoilib import *; from edge_eval import edge_dt, chamfer
K,Twc,rec,mesh,plane=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tb=np.einsum('ij,fjk->fik',Tcw,rec['T_world_mesh'].astype(float))
d=np.load('/work/scratch/masks_half.npz')
idx=[int(x) for x in sys.argv[1].split(',')]
om=(d['om'][idx]/252.).astype(np.float32); pm=d['pm'][idx]>0
rd=Renderer(mesh,K,0.5); rf=Renderer(mesh,K,1.0)
ims=read_video('/work/inputs/video.mp4',idx,0.5); imf=read_video('/work/inputs/video.mp4',idx)
valid=(1-(pm&(om<0.5))).astype(np.float32)
edt=solve.edge_dt_maps(ims,pm&(om<0.5))
f=h5py.File('/work/inputs/masks.h5')
cfg=[('D0',0,0.0),('D1',1,0.0),('D0e',0,float(sys.argv[2])),('D1e',1,float(sys.argv[2]))]
res={}
for name,dil,we in cfg:
    T,i0,i1=solve.fit_frames(rd,Tb[idx],om,valid,dil,edt=edt,w_edge=we)
    res[name]=T
res['base']=Tb[idx]
for name,T in res.items():
    ch=[];iou=[]
    for j,i in enumerate(idx):
        e,dt=edge_dt(imf[j]); v=~(cv2.dilate(f['person'][i].astype(np.uint8),np.ones((7,7),np.uint8))>0)
        ch.append(chamfer(rf,T[j],dt,e,v))
    ch=np.array(ch)
    m,_,_=rd.render(T); m=m.cpu().numpy()
    g=om>0.5; iou=((m&g)*valid).sum((1,2))/(((m|g))*valid).sum((1,2))
    print(f'{name:5s} chamfer {ch[:,0].mean():.2f}/{ch[:,1].mean():.2f}  IoU {iou.mean():.3f}  rot-vs-base {np.median(rotdeg(T[:,:3,:3],Tb[idx,:3,:3])):.2f}  rot-vs-D0 {np.median(rotdeg(T[:,:3,:3],res["D0"][:,:3,:3])):.2f} zratio {np.median(T[:,2,3]/Tb[idx,2,3]):.3f}')
