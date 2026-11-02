import sys; sys.path.insert(0,'/work/scratch/s')
from silopt import *
import time

def rot_err_deg(Ra,Rb):
    c=((np.trace(np.swapaxes(Ra,-1,-2)@Rb,axis1=-2,axis2=-1)-1)/2).clip(-1,1); return np.degrees(np.arccos(c))

def joint_fit(sf, T0, frames, data_mask, iters=300, lr=2e-3, sig_t=0.002, sig_r=0.006, lam=1.0, fixed=None, prior_w=0.0, chunk=24, verbose=True, M=None, V=None):
    """T0: (N,4,4) camera-frame init for frames; data_mask: (N,) bool which frames carry silhouette term.
    fixed: dict idx->4x4 pose that stays fixed (boundary). Smoothness: second differences of centre position and rotation matrices."""
    N=len(frames)
    if M is None: M,V=load_masks(frames)
    T0t=torch.tensor(T0,dtype=torch.float32,device='cuda')
    w=torch.zeros(N,3,device='cuda',requires_grad=True); t=torch.zeros(N,3,device='cuda',requires_grad=True)
    fixed=fixed or {}
    fidx=torch.tensor(sorted(fixed.keys()),dtype=torch.long,device='cuda')
    if len(fixed):
        Tf=torch.tensor(np.array([fixed[k] for k in sorted(fixed.keys())]),dtype=torch.float32,device='cuda')
    opt=torch.optim.Adam([w,t],lr=lr)
    didx=np.where(data_mask)[0]
    area=M.sum((1,2)).clamp_min(50)
    c=sf.c
    for it in range(iters):
        opt.zero_grad(); Ld=0
        for k in range(0,len(didx),chunk):
            ii=torch.tensor(didx[k:k+chunk],device='cuda')
            Rn,tn=sf.poses(T0t[ii],w[ii],t[ii])
            A=sf.sil(Rn,tn)
            l=((((A-M[ii])**2)*V[ii]).sum((1,2))/area[ii]).sum()
            l.backward(); Ld+=l.item()
        Rn,tn=sf.poses(T0t,w,t)
        if len(fixed):
            Rn=Rn.clone(); tn=tn.clone(); Rn[fidx]=Tf[:,:3,:3]; tn[fidx]=Tf[:,:3,3]
        p=Rn@c+tn
        d2p=p[2:]-2*p[1:-1]+p[:-2]; d2R=Rn[2:]-2*Rn[1:-1]+Rn[:-2]
        Ls=lam*((d2p**2).sum()/sig_t**2+(d2R**2).sum()/sig_r**2/2)
        Lp=prior_w*((w**2).sum()/0.02**2+(t**2).sum()/0.02**2)
        (Ls+Lp).backward()
        opt.step()
        if verbose and (it%100==0 or it==iters-1): print(f'  it {it} data {Ld:.3f} smooth {Ls.item():.3f}',flush=True)
    with torch.no_grad():
        Rn,tn=sf.poses(T0t,w,t)
        T=np.tile(np.eye(4),(N,1,1)); T[:,:3,:3]=Rn.cpu().numpy(); T[:,:3,3]=tn.cpu().numpy()
    for k,v in fixed.items(): T[k]=v
    return T

def iou_eval(sf, Tc, M, V, chunk=24):
    out=[]
    for k in range(0,len(Tc),chunk):
        T=torch.tensor(Tc[k:k+chunk],dtype=torch.float32,device='cuda')
        with torch.no_grad(): A=sf.sil(T[:,:3,:3],T[:,:3,3])>0.5
        m=M[k:k+chunk]>0.5; v=V[k:k+chunk]>0
        out+=list(((A&m&v).sum((1,2))/((A|m)&v).sum((1,2)).clamp_min(1)).cpu().numpy())
    return np.array(out)
