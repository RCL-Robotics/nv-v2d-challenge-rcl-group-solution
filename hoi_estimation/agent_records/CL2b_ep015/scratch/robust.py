import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
ids=np.arange(0,715,3); Y={}
for lam in [0.0,0.3,1.0]:
    Y[lam]=np.degrees(np.concatenate([refine_yaw(r,ev,ids[k:k+60],Tc0[ids[k:k+60]],iters=100,lam=lam,tilt_w=1.0)[4] for k in range(0,len(ids),60)]))[:,2]
segs=[(0,169),(169,375),(375,405),(405,486),(505,592),(592,715)]
for a,b in segs:
    s=(ids>=a)&(ids<b)
    print('[%d,%d) mean yaw corr: sil-only %.2f, lam0.3 %.2f, lam1 %.2f | corr(sil,lam1)=%.2f'%(a,b,Y[0.0][s].mean(),Y[0.3][s].mean(),Y[1.0][s].mean(),np.corrcoef(Y[0.0][s],Y[1.0][s])[0,1]))
