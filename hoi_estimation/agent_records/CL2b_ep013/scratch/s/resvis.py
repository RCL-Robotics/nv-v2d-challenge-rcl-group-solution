import sys; sys.path.insert(0,'/work/scratch/s')
from vis import *
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); rc=RC(mesh)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); fm=h5py.File('/work/inputs/masks.h5')
rec=np.load('/work/inputs/reconstruction.npz'); Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(float)
Tc=np.linalg.inv(Twc)[None]@np.load(sys.argv[1])['T_world_mesh']
fl=[int(x) for x in sys.argv[2].split(',')]
items=[(t,[Tcb[t],Tc[t]],'base|ours') for t in fl]
# two columns layout
vis_poses(rc,fr,fm,K,items[:len(items)//2],'/tmp/a.jpg',out=220); vis_poses(rc,fr,fm,K,items[len(items)//2:],'/tmp/b.jpg',out=220)
a=cv2.imread('/tmp/a.jpg'); b=cv2.imread('/tmp/b.jpg'); h=max(a.shape[0],b.shape[0])
a=np.pad(a,((0,h-a.shape[0]),(0,0),(0,0))); b=np.pad(b,((0,h-b.shape[0]),(0,0),(0,0)))
cv2.imwrite(sys.argv[3],np.hstack([a,np.zeros((h,10,3),np.uint8),b]))
