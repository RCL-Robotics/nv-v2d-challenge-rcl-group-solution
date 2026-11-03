import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, time
from hoilib import *
import torch.nn.functional as Fn
K,Twc,rec,mesh,plane=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tb=np.einsum('ij,fjk->fik',Tcw,rec['T_world_mesh'].astype(float))
d=np.load('masks_half.npz'); OM=d['om']; PM=d['pm']
rd=Renderer(mesh,K,0.5)
DIL=int(sys.argv[1]); F=len(Tb)
out=np.zeros((F,4,4)); ious=np.zeros((F,2))
def blur(x,k):
    if k<=1: return x
    return Fn.avg_pool2d(x[:,None],k,1,k//2,count_include_pad=False)[:,0]
def iou(s,g,v):
    s=(s>0.5).float()*v; g=(g>0.5).float()*v
    return ((s*g).sum((1,2))/((s+g-s*g).sum((1,2))+1e-6))
st=time.time()
for c0 in range(0,F,32):
    idx=list(range(c0,min(F,c0+32)))
    tgt=torch.tensor(OM[idx]/252.,device='cuda',dtype=torch.float32)
    per=torch.tensor(PM[idx],device='cuda',dtype=torch.float32)
    valid=1-per*(tgt<0.5).float()
    T0=torch.tensor(Tb[idx],device='cuda',dtype=torch.float32)
    w=torch.zeros(len(idx),3,device='cuda',requires_grad=True)
    t=torch.zeros(len(idx),3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=2e-3)
    for it in range(200):
        T=torch.cat([torch.cat([so3_exp(w)@T0[:,:3,:3],(T0[:,:3,3]+t)[...,None]],2),T0[:,3:]],1)
        sil,_,_=render_sil(rd,T)
        if DIL>0: sil=Fn.max_pool2d(sil[:,None],2*DIL+1,1,DIL)[:,0]
        k=[9,5,3,1][min(it//50,3)]
        loss=((blur(sil,k)-blur(tgt,k))**2*valid).sum((1,2))/1000
        opt.zero_grad(); loss.sum().backward(); opt.step()
        if it==0: i0=iou(sil.detach(),tgt,valid)
    i1=iou(sil.detach(),tgt,valid)
    out[idx]=T.detach().cpu().numpy(); ious[idx,0]=i0.cpu().numpy(); ious[idx,1]=i1.cpu().numpy()
print('time',time.time()-st, 'mean IoU',ious.mean(0))
np.savez(f'pf_D{DIL}.npz',T=out,iou=ious)
