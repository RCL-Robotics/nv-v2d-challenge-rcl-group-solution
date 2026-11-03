import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, time
from scipy.spatial.transform import Rotation as R
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
a,b=int(sys.argv[1]),int(sys.argv[2]); lam=float(sys.argv[3])
ids=np.arange(a,b); t0=time.time(); outs=[]
for k in range(0,len(ids),60):
    outs.append(refine_yaw(r,ev,ids[k:k+60],Tc0[ids[k:k+60]],iters=100,lam=lam,tilt_w=float(sys.argv[4])))
print('t %.1f'%(time.time()-t0))
T=np.concatenate([o[0] for o in outs]); iou=np.concatenate([o[1] for o in outs]); W=np.degrees(np.concatenate([o[4] for o in outs]))
iou0=np.concatenate([refine_yaw(r,ev,ids[k:k+60],Tc0[ids[k:k+60]],iters=0,lam=lam)[1] for k in range(0,len(ids),60)])
np.savez(f'yawref_{a}_{b}_{lam}_{sys.argv[4]}.npz',T=T,iou=iou,iou0=iou0,W=W,ids=ids)
for k in range(0,len(ids),3):
    print(ids[k],'iou %.3f -> %.3f'%(iou0[k],iou[k]),'tilt %.1f %.1f yaw %.1f'%tuple(W[k]))
