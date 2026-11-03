import sys,time; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc); D=dict(np.load('crops160.npz')); R=Renderer(v,f,uv,tex); Fi=Fitter(R,K,D,c0)
V=Tcw[None]@np.load('v1/object_poses_world.npz')['T_world_mesh']
out=V.copy()
cfg=dict(iters=int(sys.argv[1]),lr_r=float(sys.argv[2]),lr_t=float(sys.argv[3]),w_smooth=float(sys.argv[4]),w_prior=float(sys.argv[5]))
for a,b in [(129,215),(268,350),(398,481),(526,606),(640,690)]:
    fr=np.arange(a,b); t0=time.time()
    Tf,iou,ncc,h=fit_track(Fi,V[fr],fr,**cfg)
    _,iou0,ncc0,_=fit_track(Fi,V[fr],fr,iters=0)
    dang=geodesic_deg(Tf[:,:3,:3],V[fr][:,:3,:3]); dyaw=np.abs((handle_az(Tf)-handle_az(V[fr])+180)%360-180)
    print(a,b,'%.0fs'%(time.time()-t0),'iou %.3f->%.3f ncc %.3f->%.3f'%(iou0.mean(),iou.mean(),ncc0.mean(),ncc.mean()),'rot change mean %.1f max %.1f'%(dang.mean(),dang.max()))
    out[fr]=Tf
np.savez('v1_ref.npz',T_world_mesh=Twc[None]@out)
