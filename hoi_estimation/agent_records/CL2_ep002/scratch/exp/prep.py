import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tc=Tcw@rec['T_world_mesh'].astype(np.float64)
Pc=np.array([0.004,0.011,0.003])
ms=np.load('/work/scratch/mstats.npz')
anc=np.einsum('fij,j->fi',Tc[:,:3,:3],Pc)+Tc[:,:3,3]
proj=(K@anc.T).T; proj=proj[:,:2]/proj[:,2:]
diam=K[0,0]*0.26/anc[:,2]
area,bbox=ms['area'],ms['bbox']
cen=np.where((area>200)[:,None], (bbox[:,:2]+bbox[:,2:])/2, proj)
bsz=np.nan_to_num(np.max(bbox[:,2:]-bbox[:,:2],1))
size=np.maximum(2.0*diam,1.5*bsz)
obj,per,inim,boxes=load_mask_crops('/work/inputs',cen,size,128)
np.savez('/work/scratch/crops.npz',obj=obj,per=per,inim=inim,boxes=boxes)
