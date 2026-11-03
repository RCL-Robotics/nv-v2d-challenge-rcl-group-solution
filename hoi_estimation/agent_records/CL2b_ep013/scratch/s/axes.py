import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(pl[:3])
print('plane n',n)
for t in [0,60,330,360,450,500,540,660,700,860,900]:
    R=T[t,:3,:3]
    # which mesh axis is along world up (-n is up? n.x+d=0; n ~ -y => n points up since y down)
    print(t, 'n in mesh coords', np.round(R.T@n,2), 'rotvec vs 0', np.round(Rot.from_matrix(T[0,:3,:3].T@R).as_rotvec(),2), 'world axis', np.round(Rot.from_matrix(R@T[0,:3,:3].T).as_rotvec(),2))
# distance of box bottom to plane
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=np.asarray(m.vertices)[::50]
for t in [0,360,540,900]:
    pw=v@T[t,:3,:3].T+T[t,:3,3]; d=pw@n+pl[3]
    print(t,'plane dist min/max',d.min(),d.max())
