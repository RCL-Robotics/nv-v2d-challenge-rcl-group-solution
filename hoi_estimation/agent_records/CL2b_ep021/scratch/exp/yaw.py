import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs')
n=np.array(gp['plane'][:3]); 
e1=np.cross(n,[0,0,1.]); e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
def yaw_tilt(Rw):
    x=Rw[:,:,0]; y=(x@e1); z=(x@e2); yaw=np.degrees(np.arctan2(z,y))
    tilt=np.degrees(np.arccos(np.clip(-(Rw[:,:,2]@n),-1,1)))
    return yaw,tilt
Rw={'base':rec['T_world_mesh'][:,:3,:3].astype(np.float64)}
Rw['fp']=np.einsum('ij,njk->nik',Twc[:3,:3],rec['T_cam_mesh_foundationpose'][:,:3,:3].astype(np.float64))
Rw['coco']=np.einsum('ij,njk->nik',Twc[:3,:3],rec['T_cam_mesh_coconet'][:,:3,:3].astype(np.float64))
import glob
for f in sys.argv[1:]:
    d=np.load(f); Rw[f.split('/')[-1]]=np.einsum('ij,njk->nik',Twc[:3,:3],d['R'])
Y={k:yaw_tilt(v) for k,v in Rw.items()}
print('frame '+' '.join(f'{k}(yaw,tilt)' for k in Y))
for i in list(range(0,678,10)):
    print(i,' '.join('%7.1f %4.1f'%(Y[k][0][i],Y[k][1][i]) for k in Y))
