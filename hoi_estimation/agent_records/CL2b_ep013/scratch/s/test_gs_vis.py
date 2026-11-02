import sys; sys.path.insert(0,'/work/scratch/s')
from vis import *
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); rc=RC(mesh)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
rec=np.load('/work/inputs/reconstruction.npz'); res=np.load('/work/scratch/gs_test.npy',allow_pickle=True).item()
items=[]
for t,T in res.items():
    Tb=np.linalg.inv(Twc)@rec['T_world_mesh'][t]; Tn=np.linalg.inv(Twc)@T
    Tf=Tb.copy(); Tf[:3,:3]=Tb[:3,:3]@np.diag([-1,-1,1.])
    items.append((t,[Tb,Tf,Tn],'base | base flipz | search'))
vis_poses(rc,fr,f,K,items,'/work/scratch/gs_vis.jpg')
