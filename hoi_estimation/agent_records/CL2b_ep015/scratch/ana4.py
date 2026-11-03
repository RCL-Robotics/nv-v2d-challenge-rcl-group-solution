import numpy as np, json, trimesh
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
v=np.asarray(m.vertices)[::20]
for i in list(range(0,170,20))+list(range(580,715,15)):
    w=(T[i,:3,:3]@v.T).T+T[i,:3,3]; s=w@n+d   # signed dist; n points to -y (up) => positive above floor
    print(i, 'min signed dist to floor %.3f'%s.min(), 'max %.3f'%s.max())
ref=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']; s=ref@n+d; print('ref human feet min dist', s.min())
hv=r['human_vertices_world'].astype(np.float64)
print('recon human min dist per frame (sample)', [round(float((hv[i]@n+d).min()),3) for i in range(0,715,60)])
