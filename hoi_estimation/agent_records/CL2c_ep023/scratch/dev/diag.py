import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch, json
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp)
Kp=np.array(cam['K_pipeline'])
mesh=load_mesh(inp)
s=0.5
Ks=K.copy(); Ks[:2]*=s; Kps=Kp.copy(); Kps[:2]*=s
W,H=768,576
Rb=Renderer(mesh,Ks,W,H); Rp=Renderer(mesh,Kps,W,H)
F=946
Tb=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(np.float64)
Tf=rec['T_cam_mesh_foundationpose'].astype(np.float64); Tco=rec['T_cam_mesh_coconet'].astype(np.float64)
mf=h5py.File(inp+'/masks.h5')
res=np.zeros((F,3)); vis=np.zeros(F)
for a in range(0,F,32):
    ids=list(range(a,min(F,a+32)))
    om=torch.tensor(np.stack([cv2.resize(mf['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST) for i in ids])>0,device='cuda')
    pm=torch.tensor(np.stack([cv2.resize(mf['person'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST) for i in ids])>0,device='cuda')
    for j,(Rr,T) in enumerate([(Rb,Tb),(Rp,Tf),(Rp,Tco)]):
        m=Rr.render(T[ids],need_tex=False,need_depth=False)['mask']
        valid=~pm | om
        inter=(m&om&valid).sum((1,2)).float(); uni=((m|om)&valid).sum((1,2)).float()
        res[ids,j]=(inter/uni.clamp(min=1)).cpu().numpy()
    vis[ids]=om.sum((1,2)).cpu().numpy()
np.save('/work/scratch/iou_base_fp_co.npy',res)
dRf=[rot_angle_deg(Tb[i,:3,:3],Tf[i,:3,:3]) for i in range(F)]
dRc=[rot_angle_deg(Tb[i,:3,:3],Tco[i,:3,:3]) for i in range(F)]
for i in range(0,F,15):
    print(i,'iou b/fp/co %.2f %.2f %.2f'%tuple(res[i]),'vis',int(vis[i]),'dR fp %.1f co %.1f'%(dRf[i],dRc[i]), 'z b %.2f fp %.2f co %.2f'%(Tb[i,2,3],Tf[i,2,3],Tco[i,2,3]))
print('mean iou',res.mean(0))
