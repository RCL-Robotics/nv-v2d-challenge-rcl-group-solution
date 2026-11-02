import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, time
from scipy.spatial.transform import Rotation as R
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
a,b=int(sys.argv[1]),int(sys.argv[2]); lam=float(sys.argv[3]); it=int(sys.argv[4])
ids=np.arange(a,b)
t0=time.time(); out=[]
for k in range(0,len(ids),40):
    out.append(score_and_refine(r,ev,ids[k:k+40],Tc0[ids[k:k+40]],iters=it,lam=lam))
print('t',time.time()-t0)
T=np.concatenate([o[0] for o in out]); iou=np.concatenate([o[1] for o in out]); lp=np.concatenate([o[3] for o in out])
_,iou0,_,lp0=score_and_refine(r,ev,ids[:40],Tc0[ids[:40]],iters=0,lam=lam)
def jit(Tc):
    Rw=R.from_matrix(Tc[:,:3,:3]); w=(Rw[:-1].inv()*Rw[1:]).as_rotvec(); return np.degrees(np.linalg.norm(w[1:]-w[:-1],axis=1)).mean(), np.degrees(np.linalg.norm(w,axis=1)).mean()
print('baseline jitter %.2f vel %.2f'%jit(Tc0[ids]), ' refined jitter %.2f vel %.2f'%jit(T))
print('iou refined mean %.3f'%iou.mean())
d=np.array([np.degrees(np.linalg.norm(R.from_matrix(T[k,:3,:3]@Tc0[i,:3,:3].T).as_rotvec())) for k,i in enumerate(ids)])
print('mean rot change %.2f'%d.mean())
np.save(f'jit_{a}_{b}_{lam}_{it}.npy',T)
