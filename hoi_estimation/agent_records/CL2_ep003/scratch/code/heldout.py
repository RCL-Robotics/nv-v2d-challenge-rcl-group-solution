import sys,time; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc); D=dict(np.load('crops160.npz')); R=Renderer(v,f,uv,tex); Fi=Fitter(R,K,D,c0)
V=Tcw[None]@np.load('v1/object_poses_world.npz')['T_world_mesh']
cfg=dict(iters=150,lr_r=0.02,lr_t=0.002,w_smooth=float(sys.argv[1]),w_prior=float(sys.argv[2]))
tot=[]
for a,b in [(129,215),(268,350),(398,481),(526,606),(640,690)]:
    fr=np.arange(a,b); dm=(np.arange(len(fr))%2==0).astype(float)
    Tf,iou,ncc,h=fit_track(Fi,V[fr],fr,data_mask=dm,**cfg)
    _,iou0,ncc0,_=fit_track(Fi,V[fr],fr,iters=0)
    ho=dm==0
    print(a,b,'held-out iou %.3f->%.3f ncc %.3f->%.3f | train iou %.3f->%.3f'%(iou0[ho].mean(),iou[ho].mean(),ncc0[ho].mean(),ncc[ho].mean(),iou0[~ho].mean(),iou[~ho].mean()))
