import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_render import load_mesh
from hoi_fit import *
import torch.nn.functional as F
torch.manual_seed(0)
dev='cuda'
S=setup(); pm=S['pm']; par=S['par_base']; K=S['K']; Twc=S['Twc']; F_=len(S['Tb'])
vf=decimate(S['mesh'])
sr=SilRenderer(vf,K,Twc,scale=0.5)
MO,MP=load_crop_masks('/work/inputs/masks.h5',0.5)
frames=[int(x) for x in sys.argv[1].split(',')] if len(sys.argv)>1 else list(range(F_))
NH=12
res=np.zeros((F_,NH,4))
t0=time.time()
for fi in frames:
    M=torch.tensor(MO[fi],device=dev); Pm=torch.tensor(MP[fi],device=dev)
    Pm=F.max_pool2d(Pm[None,None],5,1,2)[0,0]
    valid=(1-Pm).clamp(0,1)
    th=torch.tensor(par[fi,0]+np.arange(NH)*2*np.pi/NH,dtype=torch.float32,device=dev).requires_grad_()
    a=torch.full((NH,),par[fi,1],device=dev,requires_grad=True); b=torch.full((NH,),par[fi,2],device=dev,requires_grad=True)
    opt=torch.optim.Adam([{'params':[th],'lr':0.03},{'params':[a,b],'lr':0.004}])
    for it in range(150):
        sig=max(4.0*(1-it/100),0.0)
        out=sr.render(pm.torch_pose(th,a,b,dev))
        Af,As=out[...,0],out[...,1]
        Mm=M
        if sig>0.5:
            k=int(sig*3)*2+1
            g=torch.exp(-(torch.arange(k,device=dev)-k//2)**2/(2*sig**2)); g/=g.sum()
            bl=lambda x: F.conv2d(F.conv2d(x[:,None],g[None,None,None,:],padding=(0,k//2)),g[None,None,:,None],padding=(k//2,0))[:,0]
            Af,As,Mm=bl(Af),bl(As),bl(M[None])[0]
        L=((valid*Mm*(1-Af)).sum((1,2))+(valid*(1-Mm)*As).sum((1,2)))/(valid*M).sum()
        opt.zero_grad(); L.sum().backward(); opt.step()
    res[fi,:,0]=th.detach().cpu().numpy(); res[fi,:,1]=a.detach().cpu().numpy(); res[fi,:,2]=b.detach().cpu().numpy(); res[fi,:,3]=L.detach().cpu().numpy()
    if fi%20==0: print(fi, time.time()-t0, flush=True)
np.save('/work/scratch/exp1_res.npy',res)
print('time',time.time()-t0)
