import numpy as np, json, trimesh, sys
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
gp=json.load(open('/work/inputs/ground_plane.json')); n=np.array(gp['plane'][:3]); d=gp['plane'][3]
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry(); V=m.vertices[::10]; c=m.bounds.mean(0)
Tb=np.linalg.inv(Twc)@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
fr=np.arange(262,603)
def low(Tc, anchor=None):
    out=[]
    for T in Tc:
        Tw=Twc@T; s=(V@Tw[:3,:3].T+Tw[:3,3])@n+d; out.append(s.min())
    return np.array(out)
pb=Tb[fr,:3,:3]@c+Tb[fr,:3,3]
print('base', np.round(low(Tb[fr])[::10]*100,1))
for l in sys.argv[1:]:
    T=np.load(f'move_262_602_{l}.npy').copy()
    p=T[:,:3,:3]@c+T[:,:3,3]; s=np.median(pb[:,2])/np.median(p[:,2]); T[:,:3,3]=p*s-(T[:,:3,:3]@c)
    print(l,'raw', np.round(low(np.load(f'move_262_602_{l}.npy'))[::10]*100,1))
    print(l,'anch', np.round(low(T)[::10]*100,1))
