import numpy as np, trimesh, json
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
v=np.asarray(m.vertices)
hull=m.convex_hull
# find the hull facet whose normal is closest to -z and is a support plane
best=None
for n,o in zip(hull.facets_normal, hull.facets_origin):
    if n[2]<-0.9:
        d=(v-o)@n
        print('facet normal',np.round(n,4),'area', 'max dist', d.max().round(4))
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; nf=-np.array(pl[:3])  # up normal (world -y)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float64)
for i in [0,100,160,600,650,714]:
    up=T[i,:3,:3]@np.array([0,0,1.]); print(i,'angle mesh z vs floor-up %.2f deg'%np.degrees(np.arccos(np.clip(up@-np.array(pl[:3]),-1,1))), 'vs world -y %.2f'%np.degrees(np.arccos(-up[1])))
