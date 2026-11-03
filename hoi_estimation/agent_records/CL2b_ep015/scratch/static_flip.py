import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, time
from scipy.spatial.transform import Rotation as R
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
cxy=np.array([0.005,-0.004,0])
def Rz(phi):
    T=np.eye(4); c,s=np.cos(phi),np.sin(phi); T[:2,:2]=[[c,-s],[s,c]]; T[:3,3]=cxy-T[:3,:3]@cxy; return T
for name,ids in [('start',np.arange(0,100,4)),('start_late',np.arange(110,160,4)),('end',np.arange(630,715,4))]:
    ref=Tc0[ids[0]]  # same pose for all (static): use the first frame's baseline pose
    for yaw in [0,180]:
        Tc=np.array([ref@Rz(np.radians(yaw))]*len(ids))
        for lam in [0.3]:
            T,iou,ls,lp=score_and_refine(r,ev,ids,Tc,iters=150,lam=lam,shared=True)
            _,iou2,ls2,lp2=score_and_refine(r,ev,ids,T,iters=0,lam=0.3)
            dR=np.degrees(np.linalg.norm(R.from_matrix(T[0,:3,:3]@(ref@Rz(np.radians(yaw)))[:3,:3].T).as_rotvec()))
            print(name,'yaw',yaw,'lam',lam,'iou %.3f sil %.4f photo %.3f'%(iou2.mean(),ls2.mean(),lp2.mean()),'dR from init %.1f'%dR, 'dt',np.round(T[0,:3,3]-ref[:3,3],3))
