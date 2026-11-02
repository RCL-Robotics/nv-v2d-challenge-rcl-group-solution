import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, time
import torch.nn.functional as Fnn
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
a0,a1,st=[int(x) for x in sys.argv[1:4]]
angs=np.arange(0,360,15)
def blur(x,s):
    k=int(3*s)*2+1; g=torch.exp(-((torch.arange(k,device=x.device)-k//2)**2)/(2*s*s)); g=g/g.sum()
    x=x.permute(0,3,1,2); C=x.shape[1]
    x=Fnn.conv2d(Fnn.pad(x,(k//2,k//2,0,0),mode='replicate'),g.view(1,1,1,k).repeat(C,1,1,1),groups=C)
    x=Fnn.conv2d(Fnn.pad(x,(0,0,k//2,k//2),mode='replicate'),g.view(1,1,k,1).repeat(C,1,1,1),groups=C)
    return x.permute(0,2,3,1)
out={}
for i in range(a0,a1,st):
    B=Tc[i]; box=frame_box(K,B,scale=1.3); hw=(192,192)
    img,om,pm=fd.crop(i,box,hw)
    if om.sum()<100: continue
    X=np.repeat(B[None],len(angs),0)
    for k,a in enumerate(angs): X[k,:3,:3]=B[:3,:3]@rot_about_axis([0,0,1],np.radians(a))
    Tt=torch.tensor(X,device='cuda',dtype=torch.float32)
    rgb,m,_=Rd.render(Tt,K,box,hw)
    scale=hw[0]/(box[2]-box[0])  # px of crop per image px
    O=torch.tensor(om,device='cuda'); I=torch.tensor(img,device='cuda')[None]
    mf=m.float()[...,None]
    rb=blur(rgb*mf,1.2*scale)/blur(mf,1.2*scale).clamp_min(1e-3)
    ov=(m&O).float()[...,None]; n=ov.sum((1,2,3)).clamp_min(1)
    def z(x):
        mu=(x*ov).sum((1,2))/ov.sum((1,2)).clamp_min(1); xc=(x-mu[:,None,None])*ov
        return xc/((xc**2).sum((1,2))/ov.sum((1,2)).clamp_min(1)+1e-4).sqrt()[:,None,None]
    col=((z(rb)*z(I.expand_as(rb))).sum((1,2,3))/n/3).cpu().numpy()
    # luminance only
    lum=lambda x: x.mean(-1,keepdim=True)
    coll=((z(lum(rb))*z(lum(I).expand(len(angs),-1,-1,-1))).sum((1,2,3))/n).cpu().numpy()
    out[i]=dict(col=col,lum=coll)
    kb=np.argmax(col); s=np.sort(col)
    print(i,'best %3d (%.2f) at0 %.2f at180 %.2f | lum best %3d at0 %.2f at180 %.2f'%(angs[kb],col[kb],col[0],col[12],angs[np.argmax(coll)],coll[0],coll[12]),flush=True)
np.save(f'/work/scratch/yawf_{a0}_{a1}.npy',out,allow_pickle=True)
