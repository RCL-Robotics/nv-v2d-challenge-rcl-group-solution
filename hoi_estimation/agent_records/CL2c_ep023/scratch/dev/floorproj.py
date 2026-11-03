import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp); mesh=load_mesh(inp)
v=np.asarray(mesh.vertices); n,d=plane[:3],plane[3]; g=-n
hull=trimesh.convex.convex_hull(v)
u0=np.array([-0.28,-0.02,-0.96])
k=np.argmax(hull.facets_area*(hull.facets_normal@u0>0.95)) 
fn=hull.face_normals; ang=fn@u0; areas=hull.area_faces
# use largest-area hull face within 5 deg of u0
cand=np.where(ang>np.cos(np.radians(5)))[0]; f=cand[np.argmax(areas[cand])]
dm=fn[f]; s=(v@dm).max(); print('dm',dm,'s',s)
c=0.5*(v.min(0)+v.max(0))
T=rec['T_world_mesh'].astype(np.float64); F=len(T)
o=Twc[:3,3]
def align(a,b):
    ax=np.cross(a,b); sn=np.linalg.norm(ax); cs=a@b
    if sn<1e-12: return np.eye(3)
    return Rot.from_rotvec(ax/sn*np.arctan2(sn,cs)).as_matrix()
hb=np.array([n@T[i,:3,3]+d-s for i in range(F)])  # support height under baseline? (approx, uses dm not exact)
h0=np.median(hb); print('baseline support-height median',h0, 'p10/p90',np.percentile(hb,[10,90]))
Tn=T.copy()
for i in range(F):
    Rb=T[i,:3,:3]; Rn=align(Rb@dm,g)@Rb
    pc=Rb@c+T[i,:3,3]; r=pc-o; r/=np.linalg.norm(r)
    lam=(h0+s-d-n@o+n@(Rn@c))/(n@r)
    t=o+lam*r-Rn@c; Tn[i,:3,:3]=Rn; Tn[i,:3,3]=t
np.savez('/work/scratch/floorproj.npz',T_world_mesh=Tn)
zb=(np.linalg.inv(Twc)@T)[:,2,3]; zn=(np.linalg.inv(Twc)@Tn)[:,2,3]
for i in range(0,F,20): print(i,'zb %.3f zn %.3f diff %.1fcm'%(zb[i],zn[i],100*(zn[i]-zb[i])))
print('median z',np.median(zb),np.median(zn))
dz=np.diff(zn); print('frame-to-frame |dz| median cm: base %.2f floor %.2f'%(100*np.median(np.abs(np.diff(zb))),100*np.median(np.abs(dz))))
