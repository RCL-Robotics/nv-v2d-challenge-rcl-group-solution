import numpy as np,h5py,torch,json
from rend import *
cam,r=load_all(); K=np.array(cam['K']); Kp=np.array(cam['K_pipeline']); Twc=np.array(cam['T_world_cam'])
Rn=Renderer(K)
H5=h5py.File('/work/inputs/masks.h5')
Tb=np.linalg.inv(Twc)@r['T_world_mesh'].astype(np.float64)
Tf=r['T_cam_mesh_foundationpose'].astype(np.float64); Tc=r['T_cam_mesh_coconet'].astype(np.float64)
res=np.zeros((988,3))
for i in range(988):
    om=torch.tensor(H5['object'][i],device=dev); pm=torch.tensor(H5['person'][i],device=dev)
    for j,(T,KK) in enumerate([(Tb,K),(Tf,Kp),(Tc,Kp)]):
        _,m,_=Rn.render(T[i],K=KK); m=m[0]
        valid=~pm | om
        inter=(m&om).sum(); uni=((m|om)&valid).sum()
        res[i,j]=(inter/uni.clamp(min=1)).item()
np.save('../iou_base.npy',res)
np.set_printoptions(precision=2,suppress=True,linewidth=200)
for s in range(0,988,40): print(s,res[s:s+40].mean(0))
