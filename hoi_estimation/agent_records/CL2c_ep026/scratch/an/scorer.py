import numpy as np, torch, torch.nn.functional as Fn, cv2
from render import Renderer
class Scorer:
    def __init__(self, R, frames_dir, masks, SS=2):
        self.R=R; self.fd=frames_dir; self.m=masks; self.SS=SS; self.cache={}
    def frame(self,i):
        if i not in self.cache:
            im=cv2.imread(f'{self.fd}/{i:04d}.jpg')[...,::-1].astype(np.float32)/255
            self.cache={i:(im,self.m['object'][i],self.m['person'][i])}
        return self.cache[i]
    @torch.no_grad()
    def score(self, i, Tc, crop, w_ncc=1.0, chunk=150):
        """Tc (B,4,4) camera-frame poses. returns iou(B), ncc(B)"""
        im,ob,pe=self.frame(i); x0,y0,w,h=crop
        dev=self.R.dev
        g=torch.tensor(im[y0:y0+h,x0:x0+w].mean(-1),device=dev)
        o=torch.tensor(ob[y0:y0+h,x0:x0+w],device=dev).float()
        v=1-torch.tensor(pe[y0:y0+h,x0:x0+w],device=dev).float()
        ious=[];nccs=[]
        for s in range(0,len(Tc),chunk):
            col,mk,_=self.R.render(Tc[s:s+chunk],crop=crop,res=(h*self.SS,w*self.SS))
            gr=Fn.avg_pool2d(col.mean(-1)[:,None],self.SS)[:,0]; mk=Fn.avg_pool2d(mk[:,None],self.SS)[:,0]
            inter=(mk*o*v).sum((1,2)); uni=((mk+o-mk*o)*v).sum((1,2))
            ious.append(inter/uni.clamp(min=1))
            W=(mk>0.99).float()*v; n=W.sum((1,2)).clamp(min=1)
            gm=(gr*W).sum((1,2))/n; im_m=(g*W).sum((1,2))/n
            a=(gr-gm[:,None,None])*W; b=(g-im_m[:,None,None])*W
            nccs.append((a*b).sum((1,2))/torch.sqrt((a*a).sum((1,2))*(b*b).sum((1,2))+1e-9))
        return torch.cat(ious).cpu().numpy(), torch.cat(nccs).cpu().numpy()
def crop_for(ob, Tc_proj_uv=None, pad=30, W=1536, H=1152):
    ys,xs=np.nonzero(ob)
    x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
    if Tc_proj_uv is not None:
        x0=min(x0,Tc_proj_uv[:,0].min()); x1=max(x1,Tc_proj_uv[:,0].max()); y0=min(y0,Tc_proj_uv[:,1].min()); y1=max(y1,Tc_proj_uv[:,1].max())
    x0=int(max(x0-pad,0)); y0=int(max(y0-pad,0)); x1=int(min(x1+pad,W)); y1=int(min(y1+pad,H))
    return (x0,y0,x1-x0,y1-y0)
