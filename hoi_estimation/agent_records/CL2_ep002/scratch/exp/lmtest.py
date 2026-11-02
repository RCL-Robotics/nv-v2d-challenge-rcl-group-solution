import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
import time
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
Vd,Fd=decimated_mesh(V,F); cov=CovRenderer(Vd,Fd)
Tcw=np.linalg.inv(Twc); Tc=Tcw@rec['T_world_mesh'].astype(np.float64)
Pc=np.array([0.004,0.011,0.003])
c=np.load('/work/scratch/crops.npz')
anc=np.einsum('fij,j->fi',Tc[:,:3,:3],Pc)+Tc[:,:3,3]
Kc=crop_K(K,c['boxes'],128); valid=c['inim']*(1-c['per'])
fr=np.arange(600,740)
for rep in range(2):
    t=time.time()
    R,A,iou=fit_lm(cov,Tc[fr,:3,:3],anc[fr],Kc[fr],c['obj'][fr],valid[fr],Pc,K,iters=int(sys.argv[1]))
    print(time.time()-t, iou.mean(), A[::20].round(5).tolist()[:3], R[5].ravel()[:3])
