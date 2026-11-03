import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
Vd,Fd=decimated_mesh(V,F); sil=SilRenderer(Vd,Fd)
Tcw=np.linalg.inv(Twc); Tc=Tcw@rec['T_world_mesh'].astype(np.float64)
Pc=np.array([0.004,0.011,0.003])
c=np.load('/work/scratch/crops.npz'); ms=np.load('/work/scratch/mstats.npz')
anc=np.einsum('fij,j->fi',Tc[:,:3,:3],Pc)+Tc[:,:3,3]
Kc=crop_K(K,c['boxes'],128); valid=c['inim']*(1-c['per'])
fr=np.nonzero(ms['area']>300)[0]
out=Tc.copy(); ious=np.full(980,np.nan); ious0=np.full(980,np.nan)
for s in range(0,len(fr),200):
    f=fr[s:s+200]
    R,A,iou=fit_silhouettes(sil,Tc[f,:3,:3],anc[f],Kc[f],c['obj'][f],valid[f],Pc,K,iters=300,w_rot_prior=0.02)
    _,_,iou0=fit_silhouettes(sil,Tc[f,:3,:3],anc[f],Kc[f],c['obj'][f],valid[f],Pc,K,iters=0)
    for k,i in enumerate(f): out[i,:3,:3]=R[k]; out[i,:3,3]=A[k]-R[k]@Pc
    ious[f]=iou; ious0[f]=iou0
np.savez('/work/scratch/fitA.npz',T_world_mesh=Twc@out,iou=ious,iou0=ious0)
print(np.nanmean(ious0),np.nanmean(ious))
