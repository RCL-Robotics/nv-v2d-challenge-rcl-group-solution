import sys; sys.path.insert(0,'/work/scratch/s')
from rend import *
import torch, cv2, h5py
import nvdiffrast.torch as dr

def so3_exp(w):
    th=torch.linalg.norm(w,dim=-1,keepdim=True).clamp_min(1e-9)
    k=w/th; Kx=torch.zeros(w.shape[:-1]+(3,3),device=w.device)
    Kx[...,0,1]=-k[...,2];Kx[...,0,2]=k[...,1];Kx[...,1,0]=k[...,2];Kx[...,1,2]=-k[...,0];Kx[...,2,0]=-k[...,1];Kx[...,2,1]=k[...,0]
    th=th[...,None]; I=torch.eye(3,device=w.device).expand_as(Kx)
    return I+torch.sin(th)*Kx+(1-torch.cos(th))*(Kx@Kx)

class SilFit:
    def __init__(self, r, K, H=576, W=768, scale=0.5):
        self.r=r; self.H=H; self.W=W
        Ks=np.array(K,dtype=np.float64).copy(); Ks[:2]*=scale; Ks[0,2]=(K[0,2]+0.5)*scale-0.5; Ks[1,2]=(K[1,2]+0.5)*scale-0.5
        self.K=Ks
        near,far=0.05,20.
        fx,fy,cx,cy=Ks[0,0],Ks[1,1],Ks[0,2],Ks[1,2]
        # pixel center convention: ndc = (u+0.5)/W*2-1
        self.P=torch.tensor([[2*fx/W,0,(2*cx+1)/W-1,0],[0,2*fy/H,(2*cy+1)/H-1,0],[0,0,(far+near)/(far-near),-2*far*near/(far-near)],[0,0,1,0]],dtype=torch.float32,device='cuda')
        self.c=torch.tensor(r.mesh.bounds.mean(0),dtype=torch.float32,device='cuda')
    def poses(self, T0, w, t):
        R0=T0[:,:3,:3]; t0=T0[:,:3,3]
        Rn=so3_exp(w)@R0
        tn=t0+(R0@self.c)-(Rn@self.c)+t   # rotate about mesh centre
        return Rn,tn
    def sil(self, Rn, tn):
        v=self.r.v
        vc=torch.einsum('bij,nj->bni',Rn,v)+tn[:,None]
        vh=torch.cat([vc,torch.ones_like(vc[...,:1])],-1)
        clip=torch.einsum('ij,bnj->bni',self.P,vh).contiguous()
        rast,_=dr.rasterize(self.r.glctx,clip,self.r.f,(self.H,self.W))
        m=(rast[...,3:4]>0).float()
        a=dr.antialias(m,rast,clip,self.r.f)[...,0]
        return a
def load_masks(frames, scale=0.5, dil=3):
    f=h5py.File('/work/inputs/masks.h5'); OM=f['object']; PM=f['person']
    W=int(1536*scale); H=int(1152*scale)
    om=[];val=[]
    for i in frames:
        o=cv2.resize(OM[i].astype(np.float32),(W,H),interpolation=cv2.INTER_AREA)
        p=cv2.resize(PM[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
        p=cv2.dilate(p,np.ones((2*dil+1,2*dil+1),np.uint8))>0
        om.append(o); val.append(~p)
    return torch.tensor(np.array(om),device='cuda'),torch.tensor(np.array(val),device='cuda').float()
