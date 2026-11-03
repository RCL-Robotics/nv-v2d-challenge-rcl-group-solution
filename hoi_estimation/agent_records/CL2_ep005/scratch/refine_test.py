import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, torch.nn.functional as Fn
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rot
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
om=np.load('objmask_half.npy', mmap_mode='r'); pm=np.load('permask_half.npy', mmap_mode='r')
def gblur(x,s):
    k=int(3*s)*2+1; g=torch.exp(-((torch.arange(k,device=x.device)-k//2)**2)/(2*s*s)); g=g/g.sum()
    x=Fn.conv2d(x[:,None],g[None,None,None,:],padding=(0,k//2)); x=Fn.conv2d(x,g[None,None,:,None],padding=(k//2,0)); return x[:,0]
def refine(Tc0, idx, iters=150, scale=0.5, fix_z=True, lr=2e-3):
    B=len(idx); T0=torch.tensor(Tc0,dtype=torch.float32,device='cuda')
    m=torch.tensor(np.stack([om[i] for i in idx]),device='cuda').float()
    p=torch.tensor(np.stack([pm[i] for i in idx]),device='cuda')
    valid=(~(p & (m<0.5))).float()
    w=torch.zeros(B,3,device='cuda',requires_grad=True); dxy=torch.zeros(B,2,device='cuda',requires_grad=True); dlz=torch.zeros(B,1,device='cuda',requires_grad=True)
    params=[w,dxy]+([] if fix_z else [dlz])
    opt=torch.optim.Adam(params,lr=lr)
    t0=T0[:,:3,3]; xy0=t0[:,:2]/t0[:,2:]
    mb=[gblur(m,s) for s in (4,1.5)]
    for it in range(iters):
        Rn=so3_exp(w)@T0[:,:3,:3]; z=t0[:,2:]*torch.exp(dlz); t=torch.cat([(xy0+dxy)*z,z],1)
        T=torch.zeros_like(T0); T[:,:3,:3]=Rn; T[:,:3,3]=t; T[:,3,3]=1
        sil,_,_=render_sil_soft(R,T,d['K'],d['H'],d['W'],scale)
        loss=((sil-m)**2*valid).mean((1,2))
        for s,mm in zip((4,1.5),mb): loss=loss+((gblur(sil,s)-mm)**2*valid).mean((1,2))
        opt.zero_grad(); loss.sum().backward(); opt.step()
    with torch.no_grad():
        inter=((sil>0.5)&(m>0.5)&(valid>0)).sum((1,2)).float(); uni=(((sil>0.5)|(m>0.5))&(valid>0)).sum((1,2)).float()
    return T.detach().cpu().numpy().astype(np.float64), (inter/uni).cpu().numpy()
if __name__=='__main__':
    T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
    a,b=[int(x) for x in sys.argv[1:3]]; fixz=sys.argv[3]=='1'
    idx=list(range(a,b,3))
    import time; t=time.time()
    out=[];ious=[]
    for s in range(0,len(idx),16):
        o,i=refine(Tc[idx[s:s+16]],idx[s:s+16],fix_z=fixz); out.append(o); ious.append(i)
    Tr=np.concatenate(out); io=np.concatenate(ious); print('time',time.time()-t, 'mem',torch.cuda.max_memory_allocated()/1e9)
    for name,X in [('base',Tc[idx]),('refined',Tr)]:
        Rr=Rot.from_matrix(X[:,:3,:3]); mm=Rr.mean(); ang=(Rr*mm.inv()).magnitude()*57.3
        print(name,'rot dev med/max',np.median(ang).round(2),ang.max().round(2),'t std cm',(X[:,:3,3].std(0)*100).round(2))
    print('iou refined',io.mean(), 'base', np.load('iou_base.npy')[idx].mean())
    print('rot change base->refined', np.median(rot_angle(Tc[idx][:,:3,:3],Tr[:,:3,:3])), 'z ratio', np.median(Tr[:,2,3]/Tc[idx][:,2,3]))
    np.save(f'ref_{a}_{b}_{int(fixz)}.npy',Tr)
