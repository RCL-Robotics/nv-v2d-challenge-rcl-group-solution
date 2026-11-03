import sys, numpy as np, json, h5py, torch, time
sys.path.insert(0,'/work/outputs')
from render_utils import *; from silfit import *; from traj_utils import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@P
f=h5py.File('/work/inputs/masks.h5')
a,b=201,560; fr=np.arange(a,b)
obj=f['object'][a:b]; per=f['person'][a:b]
c=R.v.mean(0).cpu().numpy()
Tsm=smooth_poses(Tc[a:b],2.0,center=c)
fit=SilFitter(R,K)
fit.setup(obj,per,Tsm)
s_base=fit.score(Tc[a:b]); s_sm=fit.score(Tsm)
t0=time.time()
ws=float(sys.argv[1]) if len(sys.argv)>1 else 0.0
Tf=fit.fit(shared=False,fix_depth=True,w_smooth=ws,iters=(80,60,60),sigmas=(6,3,1.5))
print('fit time',time.time()-t0)
s_f=fit.score(Tf)
dr=rot_angle_deg(Tf[:,:3,:3],Tsm[:,:3,:3])
cc=lambda T: np.einsum('fij,j->fi',T[:,:3,:3],c)+T[:,:3,3]
dl=np.linalg.norm(cc(Tf)-cc(Tsm),axis=1)*100
for i in range(0,b-a,10): print(a+i,'loss base %.3f smooth %.3f fit %.3f | drot %.1f dlat %.1fcm'%(s_base[i],s_sm[i],s_f[i],dr[i],dl[i]))
print('mean', s_base.mean(), s_sm.mean(), s_f.mean())
np.savez('move_fit_%g.npz'%ws, Tf=Tf, Tsm=Tsm, s_base=s_base, s_f=s_f)
