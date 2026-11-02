import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, h5py, torch, cv2
from bat_lib import *
K,Twc,rec,plane=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
T=rec['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@T
f=h5py.File('/work/inputs/masks.h5'); F=len(T)
iou=np.zeros(F); iou_vis=np.zeros(F); area_m=np.zeros(F); area_r=np.zeros(F)
masks=f['object']; pers=f['person']
for s in range(0,F,16):
    idx=list(range(s,min(F,s+16)))
    m,_=R.render(Tc[idx],K,W_IMG,H_IMG)
    m=m.cpu().numpy()>0
    om=masks[s:s+len(idx)]; pm=pers[s:s+len(idx)]
    for j,i in enumerate(idx):
        a=m[j]; b=om[j]; valid=~(pm[j]&~b)   # ignore person pixels not labeled object
        inter=(a&b).sum(); uni=((a|b)&valid).sum()
        iou[i]=inter/max(uni,1); area_m[i]=b.sum(); area_r[i]=a.sum()
np.savez('/work/scratch/base_iou.npz',iou=iou,area_m=area_m,area_r=area_r)
for i in range(0,F,10): print(i,'%.2f'%iou[i],int(area_m[i]),int(area_r[i]))
print('mean',iou.mean())
