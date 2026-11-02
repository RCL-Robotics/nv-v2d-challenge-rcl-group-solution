import sys, numpy as np, json, h5py, torch, cv2, time
sys.path.insert(0,'/work/outputs')
from render_utils import *; from silfit import *; from traj_utils import *; from edgescore import *
from scipy.spatial.transform import Rotation as Rot, Slerp
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@P
f=h5py.File('/work/inputs/masks.h5')
a,b=201,560
c=R.v.mean(0).cpu().numpy()
Tsm=smooth_poses(Tc[a:b],2.0,center=c)
ev=np.arange(a,b,2); od=np.arange(a+1,b-1,2)
imgs_ev=[cv2.imread('frames/%04d.jpg'%(i+1)) for i in ev]
fit=SilFitter(R,K); fit.setup(f['object'][a:b:2],f['person'][a:b:2],Tsm[::2],imgs=imgs_ev)
# held-out evaluator on odd frames
ho=SilFitter(R,K); ho.setup(f['object'][a+1:b-1:2],f['person'][a+1:b-1:2],Tsm[1:-1:2])
oim={i:cv2.imread('frames/%04d.jpg'%(i+1)) for i in od[::3]}
def interp(Tev):
    out=[]
    for k in range(len(od)):
        T0,T1=Tev[k],Tev[k+1]
        r=Slerp([0,1],Rot.from_matrix([T0[:3,:3],T1[:3,:3]]))(0.5).as_matrix()
        p=0.5*((T0[:3,:3]@c+T0[:3,3])+(T1[:3,:3]@c+T1[:3,3]))
        T=np.eye(4); T[:3,:3]=r; T[:3,3]=p-r@c; out.append(T)
    return np.array(out)
def evaluate(Tev,name):
    To=interp(Tev); s=ho.score(To).mean()
    e=[]
    for k,i in enumerate(od):
        if i in oim:
            _,_,dp=R.render(To[k],need_rgb=False); e.append(edge_score(dp[0].cpu().numpy(),oim[i],f['person'][i]))
    print('%-28s heldout sil %.4f edge %.3f'%(name,s,np.nanmean(e)),flush=True)
evaluate(Tc[a:b:2],'baseline')
evaluate(Tsm[::2],'baseline smoothed')
for we,ws in [(0,0),(0.05,0),(0.05,30),(0.05,300)]:
    t0=time.time()
    Tf=fit.fit(shared=False,fix_depth=True,w_edge=we,w_smooth=ws,iters=(80,60,60),sigmas=(6,3,1.5))
    evaluate(Tf,'fit we=%g ws=%g (%.0fs)'%(we,ws,time.time()-t0))
    np.save('mv2_%g_%g.npy'%(we,ws),Tf)
