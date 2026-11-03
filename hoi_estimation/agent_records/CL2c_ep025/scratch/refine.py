import numpy as np, torch, torch.nn.functional as Fn, h5py, cv2
import nvdiffrast.torch as dr
from rutil import Renderer, load_cam
dev='cuda'
def so3_exp(w):
    th=w.norm(dim=-1,keepdim=True).clamp(min=1e-8); k=w/th
    Kx=torch.zeros(w.shape[:-1]+(3,3),device=w.device,dtype=w.dtype)
    Kx[...,0,1]=-k[...,2];Kx[...,0,2]=k[...,1];Kx[...,1,0]=k[...,2];Kx[...,1,2]=-k[...,0];Kx[...,2,0]=-k[...,1];Kx[...,2,1]=k[...,0]
    th=th[...,None]; I=torch.eye(3,device=w.device,dtype=w.dtype)
    return I+torch.sin(th)*Kx+(1-torch.cos(th))*Kx@Kx
class Data:
    def __init__(self, S=192):
        mf=h5py.File('/work/inputs/masks.h5')
        self.OM=mf['object']; self.PM=mf['person']; self.S=S
        self.F=self.OM.shape[0]
        self.cache={}
    def get(self, i, fallback_uv):
        if i in self.cache: return self.cache[i]
        S=self.S
        o=self.OM[i]; ys,xs=np.nonzero(o)
        c=(xs.mean(),ys.mean()) if len(xs)>30 else fallback_uv
        x0=int(np.clip(c[0]-S/2,0,1536-S)); y0=int(np.clip(c[1]-S/2,0,1152-S))
        img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+S,x0:x0+S,::-1].astype(np.float32)/255.
        oc=o[y0:y0+S,x0:x0+S]; pc=self.PM[i][y0:y0+S,x0:x0+S]
        d=(img,oc.astype(np.float32),(~(pc&~oc)).astype(np.float32),x0,y0,len(xs))
        self.cache[i]=d; return d
class Refiner:
    def __init__(self, R, K):
        self.R=R; self.K=torch.tensor(K,dtype=torch.float32,device=dev)
    def render(self, Rm, t, x0, y0, S):
        R=self.R; B=Rm.shape[0]
        pc=torch.einsum('bij,vj->bvi',Rm,R.v)+t[:,None]
        z=pc[...,2]
        u=self.K[0,0]*pc[...,0]/z+self.K[0,2]-x0[:,None]; v=self.K[1,1]*pc[...,1]/z+self.K[1,2]-y0[:,None]
        n,f=0.05,20.
        clip=torch.stack([(2*u/S-1)*z,(2*v/S-1)*z,z*(f+n)/(f-n)-2*f*n/(f-n),z],-1).contiguous()
        rast,_=dr.rasterize(R.glctx,clip,R.f,(S,S))
        texc,_=dr.interpolate(R.uv,rast,R.f)
        col=dr.texture(R.tex,texc,filter_mode='linear')
        a=(rast[...,3:]>0).float()
        col=col*a
        out=dr.antialias(torch.cat([col,a],-1).contiguous(),rast,clip,R.f)
        return out[...,:3],out[...,3]
    def losses(self, Rm,t,img,om,valid,x0,y0,S, wcol=1.0):
        col,a=self.render(Rm,t,x0,y0,S)
        lsil=0; 
        A=a[:,None];O=om[:,None];V=valid[:,None]
        for k in [1,2,4,8]:
            if k>1:
                Ak=Fn.avg_pool2d(A,k);Ok=Fn.avg_pool2d(O,k);Vk=(Fn.avg_pool2d(V,k)>0.99).float()
            else: Ak,Ok,Vk=A,O,V
            lsil=lsil+(((Ak-Ok)**2)*Vk).sum((1,2,3))/(Ok.sum((1,2,3))+1)
        # colour on overlap
        w=(a*valid)[...,None]
        I=img; Cr=col
        # per-channel gain (detached)
        g=((I*Cr*w).sum((1,2))/((Cr*Cr*w).sum((1,2))+1e-6)).detach().clamp(0.7,1.5)
        lcol=((Cr*g[:,None,None]-I).abs()*w).sum((1,2,3))/(w.sum((1,2,3))+1)
        return lsil, lcol, a
    def optimize(self, R0, t0, img, om, valid, x0, y0, S, iters=80, wcol=1.0, lr_r=0.03, lr_t=0.01, fix_z=False):
        B=R0.shape[0]
        w=torch.zeros(B,3,device=dev,requires_grad=True); dt=torch.zeros(B,3,device=dev,requires_grad=True)
        opt=torch.optim.Adam([{'params':[w],'lr':lr_r},{'params':[dt],'lr':lr_t}])
        for it in range(iters):
            Rm=R0@so3_exp(w)  # local perturbation in object frame
            tz=dt*torch.tensor([1,1,0. if fix_z else 1],device=dev)
            t=t0+tz*t0[:,2:3]/3   # scale xy steps with depth
            ls,lc,_=self.losses(Rm,t,img,om,valid,x0,y0,S)
            loss=(ls+wcol*lc).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Rm=R0@so3_exp(w); t=t0+dt*torch.tensor([1,1,0. if fix_z else 1],device=dev)*t0[:,2:3]/3
            ls,lc,a=self.losses(Rm,t,img,om,valid,x0,y0,S)
        return Rm.detach(),t.detach(),ls,lc
