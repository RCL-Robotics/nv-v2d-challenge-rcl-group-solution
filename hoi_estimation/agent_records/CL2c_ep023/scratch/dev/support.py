import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, trimesh
from hoi_lib import *
inp='/work/inputs'
mesh=load_mesh(inp)
v=np.asarray(mesh.vertices)
hull=trimesh.convex.convex_hull(v)
u0=np.array([-0.27954436,-0.01942403,-0.9591905]); u0/=np.linalg.norm(u0)
# facets: group coplanar faces
for fi,(nrm,area) in enumerate(zip(hull.facets_normal,hull.facets_area)):
    c=nrm@u0
    if c>0.9: print(fi,nrm,area,'angle %.1f'%np.degrees(np.arccos(c)))
fn=hull.face_normals; ang=np.degrees(np.arccos(np.clip(fn@u0,-1,1)))
o=np.argsort(ang)[:10]
for k in o: print('face',k,fn[k],'%.1f'%ang[k],hull.area_faces[k])
# stability: center of mass projected inside support polygon
print('com',mesh.center_mass if mesh.is_watertight else v.mean(0))
