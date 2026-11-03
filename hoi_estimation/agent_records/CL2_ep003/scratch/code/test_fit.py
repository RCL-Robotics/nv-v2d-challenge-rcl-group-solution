import sys, time; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
cam,K,Twc,rec=load_inputs('/work/inputs')
c0,u=body_axis(v)
Tc=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(np.float64)
D=dict(np.load('crops160.npz')); R=Renderer(v,f,uv,tex)
Fi=Fitter(R,K,D,c0)
frames=[int(x) for x in sys.argv[1].split(',')]
angs=np.radians(np.arange(0,360,30))
for t in frames:
    t0=time.time()
    T0=spin(Tc[t],c0,u,angs)
    _,iou0,ncc0=Fi.fit(T0,[[t]]*len(T0),iters=0)
    Tf,iou,ncc=Fi.fit(T0,[[t]]*len(T0),iters=int(sys.argv[2]))
    print(t,'%.1fs'%(time.time()-t0))
    print('  iou0',' '.join('%2d'%(100*x) for x in iou0)); print('  ncc0',' '.join('%2d'%(100*x) for x in ncc0))
    print('  iou ',' '.join('%2d'%(100*x) for x in iou)); print('  ncc ',' '.join('%2d'%(100*x) for x in ncc))
