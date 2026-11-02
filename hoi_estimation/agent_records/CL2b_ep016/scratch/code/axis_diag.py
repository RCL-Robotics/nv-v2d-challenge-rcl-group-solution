import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, h5py
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
fp=rec['T_cam_mesh_foundationpose'].astype(float)
om=h5py.File('/work/inputs/masks.h5')['object']
F=477
def proj_axis(X):
    a=project(K,X[:3,:3]@[0,0,-0.13]+X[:3,3]); b=project(K,X[:3,:3]@[0,0,0.167]+X[:3,3])
    return b-a
out=np.full((F,6),np.nan)
for i in range(F):
    ys,xs=np.nonzero(om[i])
    if len(xs)<200: continue
    P=np.c_[xs,ys].astype(float); mu=P.mean(0); C=np.cov((P-mu).T); w,V=np.linalg.eigh(C)
    ax=V[:,1]; elong=np.sqrt(w[1]/w[0])
    pb=proj_axis(Tc[i]); 
    if ax@pb<0: ax=-ax
    ang_m=np.degrees(np.arctan2(ax[1],ax[0])); ang_b=np.degrees(np.arctan2(pb[1],pb[0]))
    pf=proj_axis(np.c_[np.r_[fp[i,:3,:3],[[0,0,0]]],np.r_[Tc[i,:3,3],1]])
    ang_f=np.degrees(np.arctan2(pf[1],pf[0]))
    # extent along axis (length in px)
    proj=(P-mu)@ax; L=np.percentile(proj,99.5)-np.percentile(proj,0.5)
    out[i]=[ang_m,ang_b,ang_f,elong,L,np.linalg.norm(pb)]
np.save('/work/scratch/axisdiag.npy',out)
for i in range(120,477,3):
    if np.isnan(out[i,0]): continue
    o=out[i]; print(i,'mask %.0f base %+.0f fp %+.0f elong %.1f Lmask %.0f Lbase %.0f'%(o[0],(o[1]-o[0]+180)%360-180,(o[2]-o[0]+180)%360-180,o[3],o[4],o[5]))
