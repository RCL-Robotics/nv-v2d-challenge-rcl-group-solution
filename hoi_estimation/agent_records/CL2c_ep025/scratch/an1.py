import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz')
cam=json.load(open('/work/inputs/camera.json')); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
Tf=r['T_cam_mesh_foundationpose'].astype(np.float64); Tc=r['T_cam_mesh_coconet'].astype(np.float64)
def ang(A,B): 
    R=np.einsum('nij,nik->njk',A[:,:3,:3],B[:,:3,:3]); return np.degrees(np.arccos(np.clip((np.trace(R,axis1=1,axis2=2)-1)/2,-1,1)))
def axang(A,B): return np.degrees(np.arccos(np.clip((A[:,:3,2]*B[:,:3,2]).sum(1),-1,1)))
db=ang(Tb[1:],Tb[:-1]); dax=axang(Tb[1:],Tb[:-1])
bf=ang(Tb,Tf); bc=ang(Tb,Tc); fc=ang(Tf,Tc)
np.set_printoptions(precision=1,suppress=True,linewidth=250)
print('frames with baseline step >15deg:', [(i+1,round(db[i],1),round(dax[i],1)) for i in np.nonzero(db>15)[0]])
for i in range(0,859,10):
    print(i, 'step%.1f'%db[min(i,857)], 'b-f %.1f b-c %.1f f-c %.1f'%(bf[i],bc[i],fc[i]), 'axis b-f %.1f b-c %.1f'%(axang(Tb[i:i+1],Tf[i:i+1])[0],axang(Tb[i:i+1],Tc[i:i+1])[0]), 'z b %.3f f %.3f c %.3f'%(Tb[i,2,3],Tf[i,2,3],Tc[i,2,3]))
