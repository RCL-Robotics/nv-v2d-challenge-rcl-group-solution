import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from bat_lib import *; from fitter import *
s=np.load('/work/scratch/simp.npz'); FR=FastRenderer(s['vs'],s['fs'],s['cs'])
K=np.array([[800.,0,768],[0,800,576],[0,0,1]])
from scipy.spatial.transform import Rotation as Ro
Rgt=Ro.from_euler('xyz',[30,70,10],degrees=True).as_matrix()
Tgt=np.eye(4); Tgt[:3,:3]=Rgt; Tgt[:3,3]=[0.1,-0.1,2.5]
win=(768-200,576-200,400.)
Kc=crop_K(K,win)
with torch.no_grad():
    S,C,_=FR.render(torch.tensor(Rgt[None],dtype=torch.float32,device='cuda'),torch.tensor(Tgt[None,:3,3],dtype=torch.float32,device='cuda'),Kc[None],RES)
M=(S[0].cpu().numpy()>0.5).astype(np.float32); I=C[0].cpu().numpy()
import cv2
D=cv2.distanceTransform((1-M).astype(np.uint8),cv2.DIST_L2,5)/RES
d=dict(M=M,P=np.zeros_like(M),D=D.astype(np.float32),I=I.astype(np.float32))
T0=Tgt.copy(); T0[:3,3]+=[0.01,0.01,0.05]; T0[:3,:3]=Rgt@Ro.from_euler('xyz',[5,5,0],degrees=True).as_matrix()
print('gt',score_batch(FR,K,Tgt[None],[win],[d],)[0],'init',score_batch(FR,K,T0[None],[win],[d])[0])
for it in [10,50,100,200]:
    T,tot,*_=fit_batch(FR,K,T0[None],[win],[d],iters=it)
    print(it,tot,T[0,:3,3]-Tgt[:3,3], rotmat_angle(T[0,:3,:3],Rgt))
