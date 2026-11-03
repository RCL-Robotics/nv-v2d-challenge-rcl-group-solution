import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, time
from hoilib import *
import torch.nn.functional as Fn
K,Twc,rec,mesh,plane=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tb=np.einsum('ij,fjk->fik',Tcw,rec['T_world_mesh'].astype(float))
d=np.load('masks_half.npz'); OM=d['om']; PM=d['pm']
rd=Renderer(mesh,K,0.5)
DIL=int(sys.argv[2])
idx=[int(x) for x in sys.argv[1].split(',')]
tgt=torch.tensor(OM[idx]/252.,device='cuda',dtype=torch.float32)
per=torch.tensor(PM[idx],device='cuda',dtype=torch.float32)
valid=1-per*(tgt<0.5).float()   # ignore person pixels that are not object
T0=torch.tensor(Tb[idx],device='cuda',dtype=torch.float32)
w=torch.zeros(len(idx),3,device='cuda',requires_grad=True)
t=torch.zeros(len(idx),3,device='cuda',requires_grad=True)
opt=torch.optim.Adam([{'params':[w],'lr':3e-3},{'params':[t],'lr':3e-3}])
def blur(x,k):
    if k<=1: return x
    return Fn.avg_pool2d(x[:,None],k,1,k//2,count_include_pad=False)[:,0]
def iou(s,g,v):
    s=(s>0.5).float()*v; g=(g>0.5).float()*v
    return ((s*g).sum((1,2))/((s+g-s*g).sum((1,2))+1e-6))
st=time.time()
for it in range(200):
    # rotate about the object centre (mesh origin approx) in camera frame
    R=so3_exp(w)
    T=T0.clone()
    T=torch.cat([torch.cat([R@T0[:,:3,:3], (T0[:,:3,3]+t)[...,None]],2), T0[:,3:]],1)
    sil,_,_=render_sil(rd,T)
    if DIL>0: sil=Fn.max_pool2d(sil[:,None],2*DIL+1,1,DIL)[:,0]
    k=[9,9,5,5,3,1][min(it//40,5)]
    loss=((blur(sil,k)-blur(tgt,k))**2*valid).sum((1,2))/1000
    opt.zero_grad(); loss.sum().backward(); opt.step()
    if it==0: i0=iou(sil.detach(),tgt,valid)
print('time',time.time()-st)
i1=iou(sil.detach(),tgt,valid)
for k,i in enumerate(idx):
    print(i,'IoU %.3f -> %.3f'%(i0[k],i1[k]),'dR %.2f deg'%np.degrees(w[k].norm().item()),'dt',np.round(t[k].detach().cpu().numpy()*100,1),'cm')
np.save('refined_test.npy',T.detach().cpu().numpy())
