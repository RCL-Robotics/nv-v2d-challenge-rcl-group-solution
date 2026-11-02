import sys,time; sys.path.insert(0,'/work/outputs')
from common import *; from fit import *
import h5py
K,Twc,_=load_camera(); r=np.load('/work/inputs/reconstruction.npz'); Tc=np.linalg.inv(Twc)@r['T_world_mesh']
Fp=r['T_cam_mesh_foundationpose'].copy(); Fp[:,:3,3]*= (Tc[:,2,3]/Fp[:,2,3])[:,None]
mesh=load_mesh()[4]; pts=sample_points(mesh,int(sys.argv[1]) if len(sys.argv)>1 else 1200)
f=h5py.File('/work/inputs/masks.h5')
def flipM(axis,ang=np.pi):
    v=np.zeros(3);v[axis]=ang;R=rotvec_to_R(v);M=np.eye(4);M[:3,:3]=R;M[:3,3]=CENTER-R@CENTER;return M
hyps=[np.eye(4),flipM(0),flipM(1),flipM(2),flipM(2,np.pi/2),flipM(2,-np.pi/2)]
F=len(Tc); NH=len(hyps)+1
out=np.zeros((F,NH,4,4)); loss=np.zeros((F,NH)); lin=np.zeros((F,NH)); lcov=np.zeros((F,NH))
t=time.time(); CH=12
for s in range(0,F,CH):
    fr=list(range(s,min(F,s+CH)))
    ob=f['object'][fr[0]:fr[-1]+1]; pe=f['person'][fr[0]:fr[-1]+1]
    data={i:FrameData(ob[k],pe[k],n_pix=800) for k,i in enumerate(fr)}
    T0=[];fid=[]
    for i in fr:
        for h in hyps: T0.append(Tc[i]@h); fid.append(i)
        T0.append(Fp[i]); fid.append(i)
    T,L=fit_poses(K,np.array(T0),fid,data,pts,iters=120)
    out[fr]=T.reshape(len(fr),NH,4,4); loss[fr]=L['loss'].reshape(len(fr),NH); lin[fr]=L['in'].reshape(len(fr),NH); lcov[fr]=L['cov'].reshape(len(fr),NH)
    if s%120==0: print(s,time.time()-t,flush=True)
areas=np.array([0]*F)
np.savez('pass1.npz',T=out,loss=loss,lin=lin,lcov=lcov)
print('total',time.time()-t)
