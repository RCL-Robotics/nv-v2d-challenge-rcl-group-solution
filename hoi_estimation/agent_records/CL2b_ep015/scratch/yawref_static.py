import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
from trajectory import chordal_mean
import pickle, time
from scipy.spatial.transform import Rotation as R
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
for a,b in [(0,169),(591,715)]:
    ids=np.arange(a,b,3)
    T,iou,ls,lp,W=refine_yaw(r,ev,ids,Tc0[ids],iters=100,lam=0.3)
    _,iou0,_,_,_=refine_yaw(r,ev,ids,Tc0[ids],iters=0,lam=0.3)
    W=np.degrees(W)
    print(a,b,'iou %.3f -> %.3f'%(iou0.mean(),iou.mean()),'tilt mean %.2f %.2f std %.2f %.2f | yaw mean %.2f std %.2f'%(W[:,0].mean(),W[:,1].mean(),W[:,0].std(),W[:,1].std(),W[:,2].mean(),W[:,2].std()))
    print('  dxy mean (cm)', np.round(100*(T[:,:2,3]-Tc0[ids,:2,3]).mean(0),2), 'std', np.round(100*(T[:,:2,3]-Tc0[ids,:2,3]).std(0),2))
