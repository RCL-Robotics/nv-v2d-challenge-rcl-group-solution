import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2
from hoi_lib import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
sc=0.25
R=Renderer(mesh,D['K'],D['W'],D['H'],scale=sc)
Tw=D['rec']['T_world_mesh'].astype(np.float64)
T=np.linalg.inv(D['T_wc'])[None]@Tw
om=read_masks('/work/inputs/masks.h5','object',sc); pm=read_masks('/work/inputs/masks.h5','person',sc)
F=len(T); res=[]
n,d=D['plane'][:3],D['plane'][3]
V=np.asarray(mesh.vertices)
for i in range(F):
    _,m,_,_=R.render(torch.tensor(T[i:i+1],dtype=torch.float32,device='cuda'),need_tex=False)
    m=m[0].cpu().numpy(); o=om[i]; valid=~(pm[i]&~o)
    inter=(m&o&valid).sum(); uni=((m|o)&valid).sum()
    vw=V@Tw[i,:3,:3].T+Tw[i,:3,3]; sd=vw@n+d
    ys,xs=np.where(o); ys2,xs2=np.where(m)
    res.append([i,inter/max(uni,1),o.sum(),m.sum(),xs.mean() if len(xs) else -1,ys.mean() if len(ys) else -1,xs2.mean(),ys2.mean(),T[i,2,3],sd.min(),sd.max(), (pm[i]&m).sum()/max(m.sum(),1)])
res=np.array(res); np.save('/work/scratch/exp/diag_base.npy',res)
np.set_printoptions(suppress=True,linewidth=200,precision=3)
for r in res[::10]: print(r)
