import sys; sys.path.insert(0,'/work/scratch/s')
from silopt import *
from scipy.spatial.transform import Rotation as Rot
exec(open('/work/scratch/s/exp_static.py').read().split('Tf=fit')[0].replace('a,b=int(sys.argv[1]),int(sys.argv[2])','a,b=int(sys.argv[1]),int(sys.argv[2])'))
c=r.mesh.bounds.mean(0)
# principal axes of mesh
_,_,vt=np.linalg.svd(r.mesh.vertices-r.mesh.vertices.mean(0),full_matrices=False)
sub=tr[::3]
res=[]
for name,ax in [('none',None),('ax0',vt[0]),('ax1',vt[1]),('ax2',vt[2])]:
    T0=Tm.copy()
    if ax is not None:
        Rf=Rot.from_rotvec(np.pi*ax).as_matrix()
        Tflip=np.eye(4); Tflip[:3,:3]=Rf; Tflip[:3,3]=c-Rf@c
        T0=Tm@Tflip
    Tf=fit(T0,sub,iters=300,lr=4e-3)
    v=iou(np.repeat(Tf[None],len(te),0),te)
    print(name,'IoU %.4f'%v.mean())
