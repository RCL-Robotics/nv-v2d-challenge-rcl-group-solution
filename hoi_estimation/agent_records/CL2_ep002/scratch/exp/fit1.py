import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
import time
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
Vd,Fd=decimated_mesh(V,F)
sil=SilRenderer(Vd,Fd)
Tcw=np.linalg.inv(Twc)
Tc=Tcw@rec['T_world_mesh'].astype(np.float64)
Pc=np.array([0.004,0.011,0.003])
c=np.load('/work/scratch/crops.npz')
a,b=int(sys.argv[1]),int(sys.argv[2]); fr=np.arange(a,b)
free=len(sys.argv)>3 and sys.argv[3]=='free'
anc=np.einsum('fij,j->fi',Tc[:,:3,:3],Pc)+Tc[:,:3,3]
Kc=crop_K(K,c['boxes'],128)
valid=c['inim']*(1-c['per'])
t=time.time()
R,A,iou=fit_silhouettes(sil,Tc[fr,:3,:3],anc[fr],Kc[fr],c['obj'][fr],valid[fr],Pc,K,iters=300,w_rot_prior=0.02,free_depth=free)
print('time',time.time()-t)
# iou of init
R0,A0,iou0=fit_silhouettes(sil,Tc[fr,:3,:3],anc[fr],Kc[fr],c['obj'][fr],valid[fr],Pc,K,iters=0)
dang=rot_angle_deg(R,Tc[fr,:3,:3])
for k in range(0,len(fr),5):
    du=(K@A[k])[:2]/A[k,2]-(K@anc[fr[k]])[:2]/anc[fr[k],2]
    print(fr[k], f'iou {iou0[k]:.2f}->{iou[k]:.2f} drot {dang[k]:5.1f} duv {du[0]:6.1f} {du[1]:6.1f} z {anc[fr[k],2]:.3f}->{A[k,2]:.3f}')
out=Tc.copy()
for k,i in enumerate(fr):
    out[i,:3,:3]=R[k]; out[i,:3,3]=A[k]-R[k]@Pc
np.savez(f'/work/scratch/fit1_{a}_{b}.npz',T_world_mesh=Twc@out)
