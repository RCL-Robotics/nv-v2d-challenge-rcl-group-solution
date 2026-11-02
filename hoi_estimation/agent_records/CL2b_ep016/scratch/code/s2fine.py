import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
import torch.nn.functional as Fnn
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
a0,a1=int(sys.argv[1]),int(sys.argv[2]); fr=list(range(a0,a1,int(sys.argv[3])))
R0=rot_mean(Tc[380:477,:3,:3]); t0=np.median(Tc[380:477,:3,3],0)
angs=np.arange(-180,180,10)
def blur(x,s):
    k=int(3*s)*2+1; g=torch.exp(-((torch.arange(k,device=x.device)-k//2)**2)/(2*s*s)); g=g/g.sum()
    x=x.permute(0,3,1,2); C=x.shape[1]
    x=Fnn.conv2d(Fnn.pad(x,(k//2,k//2,0,0),mode='replicate'),g.view(1,1,1,k).repeat(C,1,1,1),groups=C)
    x=Fnn.conv2d(Fnn.pad(x,(0,0,k//2,k//2),mode='replicate'),g.view(1,1,k,1).repeat(C,1,1,1),groups=C)
    return x.permute(0,2,3,1)
tot=np.zeros((len(angs),3))
for i in fr:
    Rb=Tc[i,:3,:3] if a1<=374 else R0
    # express yaw relative to S2 frame: rotate base about its own axis so that yaw is measured w.r.t. R0
    tb=Tc[i,:3,3] if a1<=374 else t0
    box=frame_box(K,np.r_[np.c_[Rb,tb],[[0,0,0,1]]],scale=1.3); hw=(192,192)
    img,om,pm=fd.crop(i,box,hw)
    X=np.repeat(np.eye(4)[None],len(angs),0)
    # yaw relative to R0 but keep tilt of Rb: Rb_twistfree = R0 * swing
    Q=R0.T@Rb; yawb=np.arctan2(Q[1,0],Q[0,0])
    for k,a in enumerate(angs): X[k,:3,:3]=Rb@rot_about_axis([0,0,1],np.radians(a)-yawb); X[k,:3,3]=tb
    # silhouette-only translation refine
    X,s=refine_batch(Rd,K,box,hw,X,img,om,pm,iters=40,w_col=0.0,opt_rot=False,fix_depth=True)
    rgb,m,_=Rd.render(torch.tensor(X,device='cuda',dtype=torch.float32),K,box,hw)
    sc=hw[0]/(box[2]-box[0]); mf=m.float()[...,None]
    rb=blur(rgb*mf,1.0*sc)/blur(mf,1.0*sc).clamp_min(1e-3)
    O=torch.tensor(om,device='cuda'); I=torch.tensor(img,device='cuda')[None]
    ov=(m&O).float()[...,None]
    def z(x):
        mu=(x*ov).sum((1,2))/ov.sum((1,2)).clamp_min(1); xc=(x-mu[:,None,None])*ov
        return xc/((xc**2).sum((1,2))/ov.sum((1,2)).clamp_min(1)+1e-4).sqrt()[:,None,None]
    lum=lambda x: x.mean(-1,keepdim=True)
    col=((z(lum(rb))*z(lum(I).expand(len(angs),-1,-1,-1))).sum((1,2,3))/ov.sum((1,2,3)).clamp_min(1)).cpu().numpy()
    tot+=np.c_[col,s['dice'],np.zeros_like(col)]
tot/=len(fr)
for a,(c,d,_) in zip(angs,tot): print('%4d lum %.3f dice %.3f'%(a,c,d))
