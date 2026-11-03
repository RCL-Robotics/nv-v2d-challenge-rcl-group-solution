import sys, pickle; sys.path.insert(0,'/work/outputs')
import numpy as np
from stages import *
from fitter import mirror_tilt
K,Twc,rec,plane=load_inputs('/work/inputs'); n=plane[:3]; dd=plane[3]
A=pickle.load(open('/work/scratch/stageA.pkl','rb')); res=A['res']
m=load_mesh('/work/inputs'); V=np.asarray(m.vertices); Vh=np.c_[V,np.ones(len(V))]
Tb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(np.float64)
def info(Tc,zt=None):
    T=Tc.copy()
    if zt is not None: T[:3,3]*=zt/T[2,3]
    Tw=Twc@T; ax=Tw[:3,2]; P=(Tw@Vh.T)[:3].T; h=P@n+dd
    return 'elev %.1fdeg minh %.3f maxh %.3f'%(np.degrees(np.arcsin(ax@n)),h.min(),h.max())
for i in [50,750]:
    sc=res[i]['sc']; u=unary_cost(sc)
    for j in np.argsort(u)[:8]:
        print(i,j,'u %.3f'%u[j], info(res[i]['T'][j],Tb[i,2,3]), '| own z', info(res[i]['T'][j]))
    print('base',info(Tb[i]))
