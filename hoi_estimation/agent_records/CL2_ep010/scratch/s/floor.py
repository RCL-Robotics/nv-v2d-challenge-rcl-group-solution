import numpy as np, json, trimesh
gp=json.load(open('/work/inputs/ground_plane.json')); n=np.array(gp['plane'][:3]); d=gp['plane'][3]
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
V=m.vertices[::20]
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
for i in list(range(0,794,40))+[793]:
    W=V@T[i,:3,:3].T+T[i,:3,3]; s=W@n+d
    lo=np.argsort(s)[:200]
    # tilt: angle between mesh +x (legs dir) and world down
    up_mesh=T[i,:3,:3]@np.array([1,0,0.])
    print(i, f"min {s.min()*100:6.1f}cm  p1 {np.percentile(s,1)*100:6.1f}  ang(meshx, -n) {np.degrees(np.arccos(np.clip(up_mesh@(-n),-1,1))):6.1f}")
H=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']; sh=H@n+d; print('human0 min',sh.min(), np.percentile(sh,[0.5,1,2]))
hv=np.load('/work/inputs/reconstruction.npz')['human_vertices_world']
for i in [0,100,200,400,600,700,793]:
    print('recon human',i, (hv[i]@n+d).min())
