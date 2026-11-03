import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle
from scipy.spatial.transform import Rotation as R
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
d=np.load('test_diag.npz'); Ts=d['Ts']; Tref=d['Tref']
ids=np.arange(169,592)
outs=[refine_yaw(r,ev,ids[k:k+60],Ts[ids[k:k+60]],iters=100,lam=0.3,tilt_w=1.0) for k in range(0,len(ids),60)]
T2=np.concatenate([o[0] for o in outs]); iou2=np.concatenate([o[1] for o in outs])
dy=np.degrees(np.concatenate([o[4] for o in outs]))[:,2]
d1=np.array([np.degrees(R.from_matrix(T2[k,:3,:3]@Tref[i,:3,:3].T).magnitude()) for k,i in enumerate(ids)])
print('pass2 yaw change from smoothed: mean |%.2f|, max %.1f; diff to pass1 refined obs mean %.2f'%(np.abs(dy).mean(),np.abs(dy).max(),d1.mean()))
print('IoU pass1 refined %.3f, smoothed %.3f, pass2 %.3f'%(d['iou1'][ids].mean(), np.nan, iou2.mean()))
for k in range(0,len(ids),12): print(ids[k],'dyaw %.1f iou %.3f/%.3f'%(dy[k],d['iou1'][ids[k]],iou2[k]))
