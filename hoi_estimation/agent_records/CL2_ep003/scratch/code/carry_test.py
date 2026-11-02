import sys, time; sys.path.insert(0,'/work/outputs')
from lib import *
exec(open('/work/scratch/code/upright.py').read())
sys.path.insert(0,"/work/scratch/code"); from scorelib import scores
Fi=Fitter(R,K,D,c0)
angs=np.radians(np.arange(0,360,15))
for t in [int(x) for x in sys.argv[1].split(',')]:
    T0=spin(Tc[t],c0,u,angs)
    Tf,iou,ncc=Fi.fit(T0,[[t]]*len(T0),iters=50)
    s=scores(Fi,Tf,[t])
    ang_moved=[np.degrees(np.arccos(np.clip((np.trace(T0[i,:3,:3].T@Tf[i,:3,:3])-1)/2,-1,1))) for i in range(len(Tf))]
    print(t,'moved',' '.join('%4d'%x for x in ang_moved))
    for k in s: print('  %-4s'%k,' '.join('%4d'%(100*x) for x in s[k]))
