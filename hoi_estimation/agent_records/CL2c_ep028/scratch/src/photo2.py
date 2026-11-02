import numpy as np,cv2,torch
from opt import *
from rend import Renderer
import nvdiffrast.torch as dr
SS=2
class OrRenderer(Renderer):
    def __init__(self):
        Kh=K.copy(); Kh[:2]*=SS
        super().__init__(Kh,W=W*SS,H=H*SS)
        t=(self.tex.cpu().numpy()[0]*255).astype(np.uint8)
        hsv=cv2.cvtColor(t,cv2.COLOR_RGB2HSV)
        o=((hsv[...,0]>=3)&(hsv[...,0]<=13)&(hsv[...,1]>=120)).astype(np.float32)
        self.otex=torch.tensor(np.repeat(o[...,None],3,-1),device=dev)[None].contiguous()
        self.rgbtex=self.tex
def img_orange_score(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV).astype(np.float32)
    h,s,v=hsv[...,0],hsv[...,1],hsv[...,2]
    hs=np.clip(1-np.abs(h-7.5)/5,0,1); ss=np.clip((s-90)/80,0,1); vs=np.clip((v-50)/50,0,1)
    return hs*ss*vs
ORR=None
def orange_ncc(i,Ts,im=None,min_orange=10.):
    """per hypothesis: NCC between rendered visible-orange coverage and image orange score in a window around the
    hypothesis' head bracket. nan if window leaves the image or the image shows (almost) no orange there."""
    global ORR
    if ORR is None: ORR=OrRenderer()
    if im is None: im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    out=np.full(len(Ts),np.nan)
    ORR.tex=ORR.otex
    for k,T in enumerate(Ts):
        p=T[:3,:3]@np.array([0,0,-0.475])+T[:3,3]
        if p[2]<0.3: continue
        u=K@p; c=u[:2]/u[2]
        half=int(np.clip(0.12*K[0,0]/p[2],12,80))
        x0,y0=int(c[0]-half),int(c[1]-half); s=2*half
        if x0<0 or y0<0 or x0+s>W or y0+s>H: continue
        so=img_orange_score(im[y0:y0+s,x0:x0+s])
        if so.sum()<min_orange: continue
        col,m,_=ORR.render(T,crop=(x0*SS,y0*SS,s*SS,s*SS))
        r=cv2.resize(col[0,...,0].cpu().numpy(),(s,s),interpolation=cv2.INTER_AREA)
        a=r-r.mean(); b=so-so.mean()
        out[k]=float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9))
    ORR.tex=ORR.rgbtex
    return out
if __name__=='__main__':
    d=np.load('../cands.npz')
    for i in [5,20,40,60,80,100,200,250,300,350,400,450,505,515,530,600]:
        Ts=[Rt_to_T(d['R'][i,h],d['t'][i,h]) for h in (0,6)]
        print(i,orange_ncc(i,Ts).round(3), d['L'][i,[0,6]].round(3))
