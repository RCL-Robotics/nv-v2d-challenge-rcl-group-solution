import sys,time; sys.path.insert(0,'/work/outputs')
from common import *; from fit import *
import h5py
K,Twc,_=load_camera(); r=np.load('/work/inputs/reconstruction.npz'); Tc=np.linalg.inv(Twc)@r['T_world_mesh']
mesh=load_mesh()[4]; pts=sample_points(mesh,2000)
f=h5py.File('/work/inputs/masks.h5')
def flipM(axis,ang=np.pi):
    v=np.zeros(3);v[axis]=ang;R=rotvec_to_R(v);M=np.eye(4);M[:3,:3]=R;M[:3,3]=CENTER-R@CENTER;return M
hyps=[np.eye(4),flipM(0),flipM(1),flipM(2),flipM(2,np.pi/2),flipM(2,-np.pi/2)]
frames=[0,60,150,200,300,450,600,700,780,860]
data={i:FrameData(f['object'][i],f['person'][i]) for i in frames}
T0=[];fid=[]
for i in frames:
    for h in hyps: T0.append(Tc[i]@h); fid.append(i)
t=time.time()
T,L=fit_poses(K,np.array(T0),fid,data,pts,iters=200)
print('time',time.time()-t)
L0=fit_poses(K,np.array(T0),fid,data,pts,iters=0)[1]
for k,i in enumerate(fid):
    h=k%6
    print(i,h,'init %.2f -> %.2f (in %.2f cov %.2f) rot moved %.1f z %.3f->%.3f'%(L0['loss'][k],L['loss'][k],L['in'][k],L['cov'][k],rot_angle(T[k,:3,:3],T0[k][:3,:3]),T0[k][2,3],T[k][2,3]))
np.save('t1.npy',T)
