import numpy as np, trimesh, h5py, json
m = trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
print(m, m.bounds, m.extents)
S,ext = trimesh.bounds.oriented_bounds(m); print('obb ext',ext); print(S)
r = np.load('/work/inputs/reconstruction.npz')
for k in r.files: print(k, r[k].shape, r[k].dtype)
T = r['T_world_mesh']
print(T[0]); print(T[400])
cam = json.load(open('/work/inputs/camera.json'))
Twc = np.array(cam['T_world_cam'])
Tc = np.linalg.inv(Twc)[None]@T
z = Tc[:,2,3]; print('z stats', z.min(), np.median(z), z.max())
with h5py.File('/work/inputs/masks.h5') as f:
    for k in f: print(k, f[k].shape, f[k].dtype)
    om = f['object'][:, ::4, ::4]
    pm = f['person'][:, ::4, ::4]
area = om.sum((1,2))*16
np.save('/work/scratch/obj_area.npy', area)
print('area per 20 frames', area[::20])
cp = r['contact_prob_right_left']; print('contact', np.round(cp[::20],2).T)
