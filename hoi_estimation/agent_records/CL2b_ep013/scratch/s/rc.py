import numpy as np, torch, cv2, json, trimesh, h5py
import torch.nn.functional as Fnn
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
dev='cuda'
def rotvec_to_mat(w):
    th=w.norm(dim=-1,keepdim=True).clamp(min=1e-8); k=w/th
    Kx=torch.zeros(w.shape[:-1]+(3,3),device=w.device,dtype=w.dtype)
    Kx[...,0,1]=-k[...,2];Kx[...,0,2]=k[...,1];Kx[...,1,0]=k[...,2];Kx[...,1,2]=-k[...,0];Kx[...,2,0]=-k[...,1];Kx[...,2,1]=k[...,0]
    th=th[...,None]; I=torch.eye(3,device=w.device,dtype=w.dtype).expand_as(Kx)
    return I+torch.sin(th)*Kx+(1-torch.cos(th))*Kx@Kx
class RC:
    def __init__(self, mesh, tex_scale=0.5):
        self.glctx=dr.RasterizeCudaContext()
        self.v=torch.tensor(np.asarray(mesh.vertices),dtype=torch.float32,device=dev)
        self.f=torch.tensor(np.asarray(mesh.faces),dtype=torch.int32,device=dev)
        uv=np.asarray(mesh.visual.uv).astype(np.float32).copy(); uv[:,1]=1-uv[:,1]
        self.uv=torch.tensor(uv,device=dev)
        tex=np.asarray(mesh.visual.material.baseColorTexture.convert('RGB'))
        if tex_scale!=1: tex=cv2.resize(tex,None,fx=tex_scale,fy=tex_scale,interpolation=cv2.INTER_AREA)
        self.tex=torch.tensor(tex[...,::-1].copy(),dtype=torch.float32,device=dev)[None]/255.
    def render(self, R, t, Kc, w, h, color=True):
        # R (B,3,3), t (B,3): cam_from_mesh; Kc (3,3) crop intrinsics (pixel centers at integers)
        pc=torch.einsum('bij,nj->bni',R,self.v)+t[:,None]
        Z=pc[...,2].clamp(min=1e-3)
        u=Kc[0,0]*pc[...,0]/Z+Kc[0,2]; vv=Kc[1,1]*pc[...,1]/Z+Kc[1,2]
        nx=2*(u+0.5)/w-1; ny=2*(vv+0.5)/h-1
        n_,f_=0.05,30.
        zc=Z*(f_+n_)/(f_-n_)-2*f_*n_/(f_-n_)
        cl=torch.stack([nx*Z,ny*Z,zc,Z],-1).contiguous()
        rast,_=dr.rasterize(self.glctx,cl,self.f,(h,w))
        hard=(rast[...,3:4]>0).float()
        alpha=dr.antialias(hard,rast,cl,self.f)[...,0]
        out={'alpha':alpha,'hard':hard[...,0]}
        if color:
            uvi,_=dr.interpolate(self.uv[None],rast,self.f)
            col=dr.texture(self.tex,uvi,filter_mode='linear')
            out['color']=col*hard
        d,_=dr.interpolate(pc[...,2:3].contiguous(),rast,self.f); out['depth']=d[...,0]*hard[...,0]
        return out

def crop_params(cx,cy,size,out):
    # square crop of side `size` (full-res px) centred at cx,cy resized to out px
    sc=out/size; x0=cx-size/2; y0=cy-size/2
    M=np.array([[sc,0,(-x0+0.5)*sc-0.5],[0,sc,(-y0+0.5)*sc-0.5]])
    return M,sc
def crop_K(K,M):
    Kc=K.copy().astype(np.float64); Kc[0,0]*=M[0,0]; Kc[1,1]*=M[1,1]
    Kc[0,2]=K[0,2]*M[0,0]+M[0,2]; Kc[1,2]=K[1,2]*M[1,1]+M[1,2]; return Kc
def warp(img,M,out,interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(img,M,(out,out),flags=interp,borderMode=cv2.BORDER_CONSTANT)

def gblur(x,s):
    if s<=0: return x
    k=int(2*round(3*s)+1); g=torch.exp(-(torch.arange(k,device=x.device)-k//2)**2/(2*s*s)); g=g/g.sum()
    C=x.shape[1]
    x=Fnn.conv2d(Fnn.pad(x,(k//2,k//2,0,0),mode='replicate'),g.view(1,1,1,k).repeat(C,1,1,1),groups=C)
    x=Fnn.conv2d(Fnn.pad(x,(0,0,k//2,k//2),mode='replicate'),g.view(1,1,k,1).repeat(C,1,1,1),groups=C)
    return x

class FrameObs:
    """crop of one frame: image, object mask, valid mask (not person)"""
    def __init__(self, img, om, pm, K, cx, cy, size, out=192):
        M,sc=crop_params(cx,cy,size,out); self.M=M; self.out=out
        self.Kc=torch.tensor(crop_K(K,M),dtype=torch.float32,device=dev)
        im=warp(img,M,out).astype(np.float32)/255.
        o=warp(om.astype(np.uint8)*255,M,out).astype(np.float32)/255.
        p=warp(pm.astype(np.uint8)*255,M,out).astype(np.float32)/255.
        p=cv2.dilate(p,np.ones((5,5)))
        self.img=torch.tensor(im,device=dev).permute(2,0,1)[None]
        self.om=torch.tensor(o,device=dev)[None]; self.valid=torch.tensor(1-p,device=dev)[None]
        g=cv2.cvtColor((im*255).astype(np.uint8),cv2.COLOR_BGR2GRAY)
        self.vis_img=(im*255).astype(np.uint8)

def losses(rc, obs, R, t, blur=1.0, w_sil=1.0, w_tex=1.0):
    o=rc.render(R,t,obs.Kc,obs.out,obs.out)
    a=o['alpha']; v=obs.valid; m=obs.om
    sb=max(blur,0.5)*2
    ab=gblur(a[:,None],sb)[:,0]; mb=gblur(m[:,None],sb)[:,0]
    inter=(ab*mb*v).sum((1,2)); uni=((ab+mb-ab*mb)*v).sum((1,2))
    l_sil=1-inter/uni.clamp(min=1)
    # texture: standardized blurred colour within rendered & observed & valid region
    rc_=o['color'].permute(0,3,1,2); hard=o['hard'][:,None]
    reg=(hard*v[:,None])
    # blur with normalisation by mask
    rb=gblur(rc_,blur)/gblur(hard,blur).clamp(min=1e-3)
    ib=gblur(obs.img,blur).expand_as(rb)
    n=reg.sum((2,3)).clamp(min=1)
    def stdz(x):
        mu=(x*reg).sum((2,3))/n; sd=(((x-mu[...,None,None])**2*reg).sum((2,3))/n).clamp(min=1e-6).sqrt()
        return (x-mu[...,None,None])/sd[...,None,None]
    zr=stdz(rb); zi=stdz(ib)
    ncc=((zr*zi*reg).sum((2,3))/n).mean(1)
    l_tex=1-ncc
    return w_sil*l_sil+w_tex*l_tex, l_sil, l_tex, o

def optimize(rc, obs_list, R0, t0, iters=150, lr_r=0.02, lr_t=0.0005, blurs=(3.0,2.0,1.0), w_sil=1.0, w_tex=0.0, fix_depth=False, verbose=False, z_ref=None, w_z=0.0, w_rot=0.0, w_tr=0.0, axis=None):
    """R0 (B,3,3) t0 (B,3) torch; same pose evaluated on every obs in obs_list (static object)."""
    B=R0.shape[0]
    w=torch.zeros(B,3,device=dev,requires_grad=True); d=torch.zeros(B,3,device=dev,requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr_r},{'params':[d],'lr':lr_t}])
    per=iters//len(blurs)
    for it in range(iters):
        bl=blurs[min(it//per,len(blurs)-1)]
        R=rotvec_to_mat(w)@R0
        dd=d*t0[:,2:3]  # relative translation (scale with depth)
        if fix_depth: dd=torch.cat([dd[:,:2],torch.zeros_like(dd[:,2:])],1)
        t=t0+dd
        L=0
        if z_ref is not None: L=L+w_z*((t[:,2]-z_ref)/0.05)**2
        if w_rot>0: L=L+w_rot*(w**2).sum(1)/np.radians(10)**2
        if w_tr>0: L=L+w_tr*(dd**2).sum(1)/0.05**2
        for ob in obs_list:
            wt=w_tex*min(1.0,it/(0.3*iters))
            l,_,_,_=losses(rc,ob,R,t,blur=bl,w_sil=w_sil,w_tex=wt); L=L+l
        opt.zero_grad(); L.sum().backward(); opt.step()
        if verbose and it%10==0: print(it, L.detach().cpu().numpy().round(3)[:6])
    with torch.no_grad():
        R=rotvec_to_mat(w)@R0; dd=d*t0[:,2:3]
        if fix_depth: dd=torch.cat([dd[:,:2],torch.zeros_like(dd[:,2:])],1)
        t=t0+dd
        tot=0; ls=0; lt=0
        for ob in obs_list:
            l,a,b,_=losses(rc,ob,R,t,blur=1.0,w_sil=w_sil,w_tex=w_tex); tot=tot+l; ls=ls+a; lt=lt+b
        n=len(obs_list)
    return R.detach(), t.detach(), (tot/n).cpu().numpy(), (ls/n).cpu().numpy(), (lt/n).cpu().numpy()

def uniform_rots(n, seed=0):
    return Rot.random(n, random_state=seed).as_matrix()


FLIPS=np.stack([np.eye(3),np.diag([1.,-1,-1]),np.diag([-1.,1,-1]),np.diag([-1.,-1,1])])
def score(rc, obs_list, R, t, blur=1.0):
    R=torch.as_tensor(R,dtype=torch.float32,device=dev); t=torch.as_tensor(t,dtype=torch.float32,device=dev)
    tot=0;ls=0;lt=0
    with torch.no_grad():
        for i in range(0,len(R),64):
            pass
        outs=[]
        for i in range(0,len(R),64):
            a=b=c=0
            for ob in obs_list:
                l,x,y,_=losses(rc,ob,R[i:i+64],t[i:i+64],blur=blur); a=a+l;b=b+x;c=c+y
            outs.append(torch.stack([a,b,c],1)/len(obs_list))
    return torch.cat(outs).cpu().numpy()

def global_search(rc, obs_list, t_center, n=2000, topk=24, iters=150, extra_R=None, Rs=None):
    if Rs is None: Rs=uniform_rots(n)
    if extra_R is not None: Rs=np.concatenate([extra_R,Rs])
    tt=np.broadcast_to(t_center,(len(Rs),3))
    sc=score(rc,obs_list,Rs,tt,blur=3.0)[:,1]   # silhouette only
    order=np.argsort(sc); sel=[]
    for i in order:
        if all(np.degrees(Rot.from_matrix(Rs[i].T@Rs[j]).magnitude())>15 for j in sel): sel.append(i)
        if len(sel)>=topk: break
    R0=torch.tensor(Rs[sel],dtype=torch.float32,device=dev); t0=torch.tensor(tt[sel],dtype=torch.float32,device=dev)
    R,t,_,_,_=optimize(rc,obs_list,R0,t0,iters=iters)
    R=R.cpu().numpy(); t=t.cpu().numpy()
    Rf=np.concatenate([R@F for F in FLIPS]); tf=np.concatenate([t]*4)
    s=score(rc,obs_list,Rf,tf)
    o=np.argsort(s[:,0])
    return Rf[o], tf[o], s[o]
