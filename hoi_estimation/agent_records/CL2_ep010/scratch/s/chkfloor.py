import sys; sys.path.insert(0,'/work/scratch/s')
import numpy as np, json, trimesh
from rend import load_cam
K,Kp,Twc=load_cam()
gp=json.load(open('/work/inputs/ground_plane.json')); n=np.array(gp['plane'][:3]); d=gp['plane'][3]
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry(); V=m.vertices
feet=V[V[:,0]>V[:,0].max()-0.015]
c=feet.mean(0); u,sv,vt=np.linalg.svd(feet-c); nm=vt[2]; print('feet plane normal mesh',nm, sv, len(feet))
# four foot ends
P=np.load(sys.argv[1])
for T in P:
    Tw=Twc@T
    W=feet@Tw[:3,:3].T+Tw[:3,3]; s=W@n+d
    nw=Tw[:3,:3]@nm
    print('feet height mean %.1fcm min %.1f max %.1f ; tilt %.2f deg'%(s.mean()*100,s.min()*100,s.max()*100,np.degrees(np.arccos(abs(nw@n)))))
