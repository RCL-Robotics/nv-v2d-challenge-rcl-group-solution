import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, h5py, torch, cv2, json
from bat_lib import *
K,Twc,rec,plane=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
Kp=np.array(json.load(open('/work/inputs/camera.json'))['K_pipeline'])
f=h5py.File('/work/inputs/masks.h5'); F=803
res={}
for name in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
    Tc=rec[name].astype(np.float64); iou=np.zeros(F)
    for s in range(0,F,16):
        idx=list(range(s,min(F,s+16)))
        m,_=R.render(Tc[idx],Kp,W_IMG,H_IMG); m=m.cpu().numpy()>0
        om=f['object'][s:s+len(idx)]; pm=f['person'][s:s+len(idx)]
        for j,i in enumerate(idx):
            a=m[j]; b=om[j]; valid=~(pm[j]&~b)
            iou[i]=(a&b).sum()/max(((a|b)&valid).sum(),1)
    res[name]=iou
b=np.load('/work/scratch/base_iou.npz')['iou']
np.savez('/work/scratch/iou_all.npz',base=b,fp=res['T_cam_mesh_foundationpose'],cc=res['T_cam_mesh_coconet'])
for i in range(0,F,10): print(i,'%.2f %.2f %.2f'%(b[i],res['T_cam_mesh_foundationpose'][i],res['T_cam_mesh_coconet'][i]))
print(b.mean(),res['T_cam_mesh_foundationpose'].mean(),res['T_cam_mesh_coconet'].mean())
zb=(np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(np.float64))[:,2,3]
print('z ratio base/fp', np.median(zb/rec['T_cam_mesh_foundationpose'][:,2,3]), np.median(zb/rec['T_cam_mesh_coconet'][:,2,3]))
