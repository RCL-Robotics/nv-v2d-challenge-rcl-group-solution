import sys; sys.path.insert(0,'/work/scratch/s')
from vis import *
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); rc=RC(mesh)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); fm=h5py.File('/work/inputs/masks.h5')
A=np.linalg.inv(Twc)[None]@np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
B=np.linalg.inv(Twc)[None]@np.load('/work/scratch/dev1/object_poses_world.npz')['T_world_mesh']
fl=[int(x) for x in sys.argv[1].split(',')]
vis_poses(rc,fr,fm,K,[(t,[A[t],B[t]],'final|dev') for t in fl],sys.argv[2],out=220)
