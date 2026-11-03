import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
src=sys.argv[1]
T=rec['T_world_mesh'].astype(np.float64) if src=='base' else np.load(src)['T_world_mesh']
Tc=Tcw@T
m=h5py.File('/work/inputs/masks.h5','r'); O=m['object']; P=m['person']
out=[]
for i in range(980):
    om=O[i]; pm=P[i]
    _,mk,_=R.render(Tc[i:i+1]); mk=mk[0].cpu().numpy()
    v=~pm
    inter=(mk&om&v).sum(); uni=((mk|om)&v).sum()
    iou=inter/uni if uni>0 else np.nan
    ys,xs=np.nonzero(mk&v); yo,xo=np.nonzero(om)
    d=(xs.mean()-xo.mean(), ys.mean()-yo.mean()) if len(xs) and len(xo) else (np.nan,np.nan)
    out.append([iou,(mk&v).sum(),om.sum(),d[0],d[1]])
out=np.array(out); np.save(sys.argv[2],out)
for a in range(0,980,20):
    s=out[a:a+20]; print(a, np.nanmean(s,0).round(2))
