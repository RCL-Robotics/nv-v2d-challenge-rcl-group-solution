import sys, numpy as np, json, h5py, torch, cv2
sys.path.insert(0,'/work/outputs')
from render_utils import *; from silfit import *; from traj_utils import *; from edgescore import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@P
f=h5py.File('/work/inputs/masks.h5')
a,b=int(sys.argv[1]),int(sys.argv[2]); st=4; we=float(sys.argv[3])
fr=np.arange(a,b,st); obj=f['object'][a:b:st]; per=f['person'][a:b:st]
imgs=[cv2.imread('frames/%04d.jpg'%(i+1)) for i in fr]
c=np.load('static2_%d.npz'%a); Tm=c['mean']; rm=Tm[:3,:3]
fit=SilFitter(R,K); fit.setup(obj,per,np.repeat(Tm[None],len(fr),0),imgs=imgs)
efr=np.arange(a+1,b,10); eim={i:cv2.imread('frames/%04d.jpg'%(i+1)) for i in efr}
def es(T):
    o=[]
    for i in efr:
        _,_,dp=R.render(T,need_rgb=False); o.append(edge_score(dp[0].cpu().numpy(),eim[i],f['person'][i]))
    return np.nanmean(o)
res={}
for nm,kw in [('A',dict(iters=(60,60,60),sigmas=(4,2,1))),('B',dict(iters=(80,60,60),sigmas=(6,3,1.5))),('C',dict(iters=(100,100,100,100),sigmas=(6,3,1.5,0.75),lr_r=0.005,lr_t=0.003))]:
    T=fit.fit(shared=True,fix_depth=True,w_edge=we,**kw)[0]; res[nm]=T
    print(nm,'w_edge',we,'sil %.4f edge(heldout frames) %.3f'%(fit.score(np.repeat(T[None],len(fr),0)).mean(), es(T)), 'rot vs mean %.2f'%rot_angle_deg(T[:3,:3],rm))
ks=list(res); print('pairwise rot diffs', [(ks[i],ks[j],round(float(rot_angle_deg(res[ks[i]][:3,:3],res[ks[j]][:3,:3])),2)) for i in range(3) for j in range(i+1,3)])
np.savez('static3_%d_%g.npz'%(a,we),**res)
