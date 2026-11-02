import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, time
from hoilib import *
import torch.nn.functional as Fn
K,Twc,rec,mesh,plane=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tb=np.einsum('ij,fjk->fik',Tcw,rec['T_world_mesh'].astype(float))
d=np.load('masks_half.npz'); OM=d['om']; PM=d['pm']
rd=Renderer(mesh,K,0.5)
a,b,step,DIL=[int(x) for x in sys.argv[1:5]]
fixz=len(sys.argv)>5
idx=list(range(a,b,step))
T0np=robust_mean_pose(Tb[a:b]); print('baseline rot spread (deg) median',np.median(rotdeg(T0np[None,:3,:3],Tb[a:b,:3,:3])))
tgt=torch.tensor(OM[idx]/252.,device='cuda',dtype=torch.float32)
per=torch.tensor(PM[idx],device='cuda',dtype=torch.float32)
valid=1-per*(tgt<0.5).float()
T0=torch.tensor(T0np,device='cuda',dtype=torch.float32)
w=torch.zeros(1,3,device='cuda',requires_grad=True)
t=torch.zeros(1,3,device='cuda',requires_grad=True)
opt=torch.optim.Adam([{'params':[w],'lr':2e-3},{'params':[t],'lr':2e-3}])
def blur(x,k):
    if k<=1: return x
    return Fn.avg_pool2d(x[:,None],k,1,k//2,count_include_pad=False)[:,0]
def iou(s,g,v):
    s=(s>0.5).float()*v; g=(g>0.5).float()*v
    return ((s*g).sum((1,2))/((s+g-s*g).sum((1,2))+1e-6))
def make(w,t):
    R=so3_exp(w)[0]
    tt=t[0].clone()
    if fixz: tt=torch.stack([tt[0],tt[1],tt[2]*0])
    T=torch.eye(4,device='cuda'); T=T.clone()
    T=torch.cat([torch.cat([R@T0[:3,:3],(T0[:3,3]+tt)[:,None]],1),T0[3:]],0)
    return T
for it in range(150):
    T=make(w,t)
    sil,_,_=render_sil(rd,T[None])
    if DIL>0: sil=Fn.max_pool2d(sil[:,None],2*DIL+1,1,DIL)[:,0]
    sil=sil.expand(len(idx),-1,-1)
    k=[9,5,3,1][min(it//40,3)]
    loss=((blur(sil,k)-blur(tgt,k))**2*valid).sum()/1000/len(idx)
    opt.zero_grad(); loss.backward(); opt.step()
    if it==0: i0=iou(sil.detach(),tgt,valid)
i1=iou(sil.detach(),tgt,valid)
print(f'phase {a}-{b} DIL{DIL} fixz{fixz}: IoU mean {i0.mean():.3f} -> {i1.mean():.3f}; dR {np.degrees(w.norm().item()):.2f} deg; dt(cm) {np.round(t[0].detach().cpu().numpy()*100,1)}')
np.save(f'static_{a}_{b}_{DIL}_{int(fixz)}.npy',T.detach().cpu().numpy())
