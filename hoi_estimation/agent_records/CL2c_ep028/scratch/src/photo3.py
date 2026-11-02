import numpy as np,cv2,torch
from opt import *
from rend import Renderer
import nvdiffrast.torch as dr
SS=2
class OrRenderer(Renderer):
    def __init__(self):
        super().__init__(K)
        t=(self.tex.cpu().numpy()[0]*255).astype(np.uint8)
        hsv=cv2.cvtColor(t,cv2.COLOR_RGB2HSV)
        o=((hsv[...,0]>=3)&(hsv[...,0]<=13)&(hsv[...,1]>=120)).astype(np.float32)
        self.otex=torch.tensor(o[...,None],device=dev)[None].contiguous()
    def render_crops(self,T,x0,y0,s):
        """T (B,4,4) cam poses, per-item crop origins x0,y0 (full-res px), common size s -> orange coverage (B,s,s) at full res"""
        B=len(T); S=s*SS
        fx,fy,cx,cy=K[0,0]*SS,K[1,1]*SS,K[0,2]*SS,K[1,2]*SS
        n,fa=0.05,20.
        P=torch.zeros(B,4,4,device=dev)
        P[:,0,0]=2*fx/S; P[:,0,2]=torch.tensor(2*(cx-x0*SS)/S-1,dtype=torch.float32,device=dev)
        P[:,1,1]=2*fy/S; P[:,1,2]=torch.tensor(2*(cy-y0*SS)/S-1,dtype=torch.float32,device=dev)
        P[:,2,2]=(fa+n)/(fa-n); P[:,2,3]=-2*fa*n/(fa-n); P[:,3,2]=1
        Tt=torch.tensor(T,dtype=torch.float32,device=dev)
        vh=torch.cat([self.v,torch.ones_like(self.v[:,:1])],1)
        clip=torch.einsum('bij,bjk,nk->bni',P,Tt,vh)*torch.tensor([1,-1,1,1],device=dev,dtype=torch.float32)
        rast,_=dr.rasterize(self.glctx,clip.contiguous(),self.f,(S,S))
        rast=torch.flip(rast,[1])
        texc,_=dr.interpolate(self.uv[None],rast,self.f)
        col=dr.texture(self.otex,texc,filter_mode='linear')[...,0]*(rast[...,3]>0)
        return torch.nn.functional.avg_pool2d(col[:,None],SS)[:,0]
def img_orange_score(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV).astype(np.float32)
    h,s,v=hsv[...,0],hsv[...,1],hsv[...,2]
    return np.clip(1-np.abs(h-7.5)/5,0,1)*np.clip((s-90)/80,0,1)*np.clip((v-50)/50,0,1)
ORR=None
def orange_ncc_batch(Ts,im,min_orange=10.):
    global ORR
    if ORR is None: ORR=OrRenderer()
    Ts=np.asarray(Ts); out=np.full(len(Ts),np.nan)
    p=np.einsum('bij,j->bi',Ts[:,:3,:3],np.array([0,0,-0.475]))+Ts[:,:3,3]
    ok=p[:,2]>0.3
    if not ok.any(): return out
    zc=np.median(p[ok,2]); half=int(np.clip(0.12*K[0,0]/zc,12,80)); s=2*half
    u=(K@p.T).T; c=u[:,:2]/np.maximum(u[:,2:],1e-3)
    x0=(c[:,0]-half).astype(int); y0=(c[:,1]-half).astype(int)
    ok&=(x0>=0)&(y0>=0)&(x0+s<=W)&(y0+s<=H)
    idx=np.nonzero(ok)[0]
    if len(idx)==0: return out
    so=np.stack([img_orange_score(im[y0[k]:y0[k]+s,x0[k]:x0[k]+s]) for k in idx])
    r=ORR.render_crops(Ts[idx],x0[idx],y0[idx],s).cpu().numpy()
    for j,k in enumerate(idx):
        if so[j].sum()<min_orange: continue
        a=r[j]-r[j].mean(); b=so[j]-so[j].mean()
        out[k]=float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9))
    return out
if __name__=='__main__':
    import time,sys
    d=np.load('../cands2.npz'); N=np.load('../ncc2.npy')
    cap=cv2.VideoCapture('/work/inputs/video.mp4'); t0=time.time(); M=[]
    for i in range(200):
        ok,im=cap.read()
        Ts=Rt_to_T(d['R'][i],d['t'][i]); M.append(orange_ncc_batch(Ts,im))
    print(time.time()-t0); M=np.array(M); print(np.nanmean(np.abs(M-N[:200])), np.isnan(M).mean(), np.isnan(N[:200]).mean())
