import sys, numpy as np, json, h5py, torch, cv2
sys.path.insert(0,'/work/outputs')
from render_utils import *; from silfit import *; from traj_utils import *; from edgescore import *
from scipy.spatial.transform import Rotation as Rot
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@P
f=h5py.File('/work/inputs/masks.h5')
a,b=int(sys.argv[1]),int(sys.argv[2]); st=2
fr=np.arange(a,b,st); obj=f['object'][a:b:st]; per=f['person'][a:b:st]
rm=rot_mean(Tc[a:b,:3,:3]); tm=np.median(Tc[a:b,:3,3],0)
Tm=np.eye(4); Tm[:3,:3]=rm; Tm[:3,3]=tm
fit=SilFitter(R,K); fit.setup(obj,per,np.repeat(Tm[None],len(fr),0))
cands={'mean':Tm}
for nm,kw in [('A',dict(iters=(60,60,60),sigmas=(4,2,1))),('B',dict(iters=(80,60,60),sigmas=(6,3,1.5))),('C',dict(iters=(100,100,100,100),sigmas=(6,3,1.5,0.75),lr_r=0.005,lr_t=0.003))]:
    cands[nm]=fit.fit(shared=True,fix_depth=True,**kw)[0]
efr=np.arange(a,b,10)
imgs={i:cv2.imread('frames/%04d.jpg'%(i+1)) for i in efr}
def es(Tlist):
    o=[]
    for T,i in zip(Tlist,efr):
        _,_,dp=R.render(T,need_rgb=False); o.append(edge_score(dp[0].cpu().numpy(),imgs[i],f['person'][i]))
    return np.nanmean(o)
print('baseline per-frame: sil %.4f edge %.3f'%(fit.score(Tc[fr]).mean(), es(Tc[efr])))
for k,T in cands.items():
    print(k,'sil %.4f edge %.3f'%(fit.score(np.repeat(T[None],len(fr),0)).mean(), es([T]*len(efr))), 'rot vs mean %.2f'%rot_angle_deg(T[:3,:3],rm))
np.savez('static2_%d.npz'%a, **cands)
