import sys; sys.path.insert(0,'/work/scratch/s')
from vis import *
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); rc=RC(mesh)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); fm=h5py.File('/work/inputs/masks.h5')
rec=np.load('/work/inputs/reconstruction.npz'); Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(float)
d=np.load(sys.argv[1],allow_pickle=True).item()
a,b,s=[int(x) for x in sys.argv[2:5]]
keys=sys.argv[6].split(',') if len(sys.argv)>6 else ['fw','bw']
items=[(t,[Tcb[t]]+[d[k][t][0] if isinstance(d[k][t],tuple) else d[k][t] for k in keys],'base|'+'|'.join(keys)) for t in range(a,b+1,s)]
vis_poses(rc,fr,fm,K,items,sys.argv[5],out=200)
