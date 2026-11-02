import sys, pickle; sys.path.insert(0,'/work/outputs')
import numpy as np
from stages import *
from fitter import mirror_tilt
K,Twc,rec,plane=load_inputs('/work/inputs'); Tcw=np.linalg.inv(Twc)
A=pickle.load(open('/work/scratch/stageA.pkl','rb')); res=A['res']
Tb=Tcw[None]@rec['T_world_mesh'].astype(np.float64)
kp=rec['human_keypoints_world'].astype(np.float64)
kpc=np.einsum('ij,fkj->fki',Tcw[:3,:3],kp)+Tcw[:3,3]
hands=np.r_[21:63]
zs=np.linspace(-0.2,0.33,30); seg=np.stack([np.zeros(30),np.zeros(30),zs,np.ones(30)],1)
def hdist(T,i,zt):
    T=T.copy(); T[:3,3]*=zt/T[2,3]; P=(T@seg.T)[:3].T
    H=kpc[i,hands]
    D=np.linalg.norm(H[:,None]-P[None],axis=-1)
    return D.min()   # closest hand keypoint to bat axis
dec=np.load('/work/scratch/dec1.npz')
out=[]
for i in range(115,700,5):
    T=dec['Tc'][i]; Tm=mirror_tilt(T)
    a=hdist(T,i,Tb[i,2,3]); b=hdist(Tm,i,Tb[i,2,3])
    ax=T[:3,2]; r=T[:3,3]/np.linalg.norm(T[:3,3]); tilt=np.degrees(np.arcsin(abs(ax@r)))
    out.append((i,a,b,tilt))
    print(i,'chosen %.3f mirror %.3f  tilt %.0f'%(a,b,tilt))
