import sys,time; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc)
D=dict(np.load('crops160.npz')); R=Renderer(v,f,uv,tex); Fi=Fitter(R,K,D,c0)
P=np.load('pred_handrigid.npz')['T_world_mesh']; B=rec['T_world_mesh'].astype(np.float64)
a,b=int(sys.argv[1]),int(sys.argv[2]); fr=np.arange(a,b)
for name,TW in [('hand',P),('base',B)]:
    T0=Tcw[None]@TW[a:b]
    t0=time.time()
    Tf,iou,ncc,h=fit_track(Fi,T0,fr,iters=int(sys.argv[3]),fix_ends=False)
    _,iou0,ncc0,_=fit_track(Fi,T0,fr,iters=0)
    print(name,'%.1fs'%(time.time()-t0),'iou0 %.3f ncc0 %.3f -> iou %.3f ncc %.3f'%(iou0.mean(),ncc0.mean(),iou.mean(),ncc.mean()))
    print('  per5 iou',np.round(iou[::5]*100).astype(int))
    np.save(f'cr_{name}_{a}.npy',Tf)
