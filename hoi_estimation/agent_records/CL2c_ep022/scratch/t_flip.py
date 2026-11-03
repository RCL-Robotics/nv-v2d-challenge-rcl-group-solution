import sys, numpy as np, json, h5py, torch, trimesh
sys.path.insert(0,'/work/outputs')
from render_utils import *; from silfit import *
from scipy.spatial.transform import Rotation as Rot
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
S_,_=trimesh.bounds.oriented_bounds(m); A=np.linalg.inv(S_)[:3,:3]  # columns = OBB axes in mesh coords
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@P
f=h5py.File('/work/inputs/masks.h5')
a,b=int(sys.argv[1]),int(sys.argv[2]); fr=np.arange(a,b,8)
obj=f['object'][a:b:8]; per=f['person'][a:b:8]
rm=Rot.from_matrix(Tc[a:b,:3,:3]).mean().as_matrix(); tm=np.median(Tc[a:b,:3,3],0)
c=R.v.mean(0).cpu().numpy()
fit=SilFitter(R,K)
for k in range(4):
    F_=np.eye(3) if k==0 else Rot.from_rotvec(np.pi*A[:,k-1]).as_matrix()
    Rn=rm@F_; t=rm@c+tm-Rn@c   # flip about object centre
    T=np.eye(4); T[:3,:3]=Rn; T[:3,3]=t
    fit.setup(obj,per,np.repeat(T[None],len(fr),0))
    s0=fit.score(np.repeat(T[None],len(fr),0)).mean()
    Tf=fit.fit(shared=True,fix_depth=True,iters=(80,60,60),sigmas=(6,3,1.5))
    print('flip axis',k,'init %.3f fitted %.3f'%(s0,fit.score(Tf).mean()), 'rot from base %.1f'%np.degrees(Rot.from_matrix(Tf[0,:3,:3]@rm.T).magnitude()))
