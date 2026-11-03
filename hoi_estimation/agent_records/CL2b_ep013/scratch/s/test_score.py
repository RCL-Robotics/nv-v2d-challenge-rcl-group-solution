import sys; sys.path.insert(0,'/work/scratch/s')
from rc import *
import json
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
rc=RC(mesh)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
rec=np.load('/work/inputs/reconstruction.npz')
for t in [int(x) for x in sys.argv[1].split(',')]:
    Tc=np.linalg.inv(Twc)@rec['T_world_mesh'][t]
    om=f['object'][t]; ys,xs=np.nonzero(om)
    obs=FrameObs(fr[t],om,f['person'][t],K,xs.mean(),ys.mean(),max(xs.ptp(),ys.ptp())*2.0)
    R0=Tc[:3,:3]; flips=[np.eye(3),np.diag([1,-1,-1]),np.diag([-1,1,-1]),np.diag([-1,-1,1])]
    R=torch.tensor(np.stack([R0@F for F in flips]),dtype=torch.float32,device=dev)
    tt=torch.tensor(np.stack([Tc[:3,3]]*4),dtype=torch.float32,device=dev)
    L,ls,lt,o=losses(rc,obs,R,tt)
    print(t,'total',L.cpu().numpy().round(3),'sil',ls.cpu().numpy().round(3),'tex',lt.cpu().numpy().round(3))
