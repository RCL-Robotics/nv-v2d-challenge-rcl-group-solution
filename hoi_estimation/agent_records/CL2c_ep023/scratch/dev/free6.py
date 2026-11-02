import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, h5py
from solve import *
import solve
cam,K,Twc,rec,plane=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); mv=np.asarray(mesh.vertices)
Tcw=np.linalg.inv(Twc)
T1=np.load('/work/scratch/v1.npz')['T_world_mesh']
ids=list(range(0,135,3))+list(range(880,946,3))
with h5py.File('/work/inputs/masks.h5') as mf: M,V,Kc=build_crops(mf,ids,Tcw@T1[ids],K,mv)
R=Renderer(mesh); dev='cuda'
Tc0=torch.tensor(Tcw@T1[ids],dtype=torch.float32,device=dev)
def so3(w):
    th=w.norm(dim=1,keepdim=True).clamp(min=1e-8)[:,:,None]; k=w/w.norm(dim=1,keepdim=True).clamp(min=1e-8)
    Kx=torch.zeros(len(w),3,3,device=dev); Kx[:,0,1]=-k[:,2];Kx[:,0,2]=k[:,1];Kx[:,1,0]=k[:,2];Kx[:,1,2]=-k[:,0];Kx[:,2,0]=-k[:,1];Kx[:,2,1]=k[:,0]
    return torch.eye(3,device=dev)+torch.sin(th)*Kx+(1-torch.cos(th))*Kx@Kx
w=torch.zeros(len(ids),3,device=dev,requires_grad=True); t=torch.zeros(len(ids),3,device=dev,requires_grad=True)
opt=torch.optim.Adam([{'params':[w],'lr':0.005},{'params':[t],'lr':0.003}])
Mt=torch.tensor(M,device=dev);Vt=torch.tensor(V,device=dev);Kt=torch.tensor(Kc,device=dev)
c=torch.tensor(0.5*(mv.min(0)+mv.max(0)),dtype=torch.float32,device=dev)
for it in range(200):
    Rd=so3(w); Tc=Tc0.clone(); 
    # rotate about object centre in camera frame
    ctr=(Tc0[:,:3,:3]@c)+Tc0[:,:3,3]
    Tc=torch.zeros_like(Tc0); Tc[:,3,3]=1; Tc[:,:3,:3]=Rd@Tc0[:,:3,:3]; Tc[:,:3,3]=Rd@(Tc0[:,:3,3]-ctr)[:,:,None][...,0]+ctr+t if False else (torch.einsum('bij,bj->bi',Rd,Tc0[:,:3,3]-ctr)+ctr+t)
    al=render_alpha(R,Tc,Kt,RES); al=torch.nn.functional.max_pool2d(al[:,None],7,1,3)[:,0]
    L=1-(al*Mt*Vt).sum((1,2))/((al+Mt-al*Mt)*Vt).sum((1,2)).clamp(min=1)
    opt.zero_grad(); L.sum().backward(); opt.step()
Tf=Tc.detach().cpu().numpy().astype(np.float64); Tw=Twc@Tf
fm=FloorModel(mesh,plane,np.array([-0.25,0,-0.97]))
g=fm.g
for k,(a,b) in enumerate([(0,45),(45,len(ids))]):
    tilt=[np.degrees(np.arccos(np.clip((T[:3,:3]@fm.dm)@g,-1,1))) for T in Tw[a:b]]
    dz=100*(Tf[a:b,2,3]-(Tcw@T1[ids])[a:b,2,3]); drot=[rot_angle_deg(Tw[i,:3,:3],T1[ids[i],:3,:3]) for i in range(a,b)]
    print('phase',k,'loss %.3f'%L[a:b].mean().item(),'tilt med %.2f (std %.2f)'%(np.median(tilt),np.std(tilt)),'rot change med %.2f'%np.median(drot),'dz med %.1fcm std %.1f'%(np.median(dz),np.std(dz)))
