import sys, numpy as np, json, torch, h5py
sys.path.insert(0,'/work/outputs')
from render_utils import *
from scipy.spatial.transform import Rotation as Rot
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K,scale=0.5)
src=sys.argv[1]
poses=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'] if src=='base' else np.load(src)['T_world_mesh']
f=h5py.File('/work/inputs/masks.h5')
F=len(poses); iou=np.zeros(F); area=np.zeros(F); ra=np.zeros(F)
for s in range(0,F,32):
    idx=list(range(s,min(F,s+32)))
    T=np.linalg.inv(Twc)[None]@poses[idx]
    _,mk,_=R.render(T,need_rgb=False); mk=mk.cpu().numpy()
    mo=f['object'][s:s+len(idx)][:,::2,::2]; mp=f['person'][s:s+len(idx)][:,::2,::2]
    for j,i in enumerate(idx):
        valid=~(mp[j]&~mo[j])
        a=mk[j]&valid; b=mo[j]
        iou[i]=(a&b).sum()/max(1,(a|b).sum()); area[i]=b.sum(); ra[i]=mk[j].sum()
Tc=np.linalg.inv(Twc)[None]@poses
rot=Rot.from_matrix(poses[:,:3,:3])
dr_=np.degrees((rot[1:]*rot[:-1].inv()).magnitude())
dt=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)*100
np.savez(sys.argv[2],iou=iou,area=area,ra=ra,dr=dr_,dt=dt,z=Tc[:,2,3])
for i in range(0,F,10):
    print(i, 'iou %.2f area %6d rarea %6d z %.2f dr %.1f dt %.1f'%(iou[i],area[i],ra[i],Tc[i,2,3],dr_[min(i,F-2)],dt[min(i,F-2)]))
print('mean iou',iou.mean())
big=np.argsort(-dr_)[:30]; print('largest rot jumps', sorted([(int(b),round(float(dr_[b]),1)) for b in big]))
