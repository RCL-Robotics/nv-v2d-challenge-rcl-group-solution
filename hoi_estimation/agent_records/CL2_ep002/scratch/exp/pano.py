import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
import cv2, torch, nvdiffrast.torch as dr
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
AXC=np.array([0.004,0.0,0.003])
# per-face attributes (mesh is unshared: 3 verts per face?)
fn=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]); fn/=np.linalg.norm(fn,axis=1,keepdims=True)+1e-12
fc=V[F].mean(1); rad=fc[:,[0,2]]-AXC[[0,2]]; r=np.linalg.norm(rad,axis=1); rad/=r[:,None]+1e-12
radial=(fn[:,0]*rad[:,0]+fn[:,2]*rad[:,1])
y=fc[:,1]; rb=0.089+(y+0.067)/0.156*(0.114-0.089)
body=(np.abs(r-rb)<0.006)&(y>-0.06)&(y<0.085)
outer=(radial>0.5)&body
print('outer faces',outer.sum(),'of',len(F))
# unshare vertices for per-face attributes
Vu=V[F].reshape(-1,3); Fu=np.arange(len(Vu)).reshape(-1,3)
phi=np.arctan2(Vu[:,2]-AXC[2],Vu[:,0]-AXC[0])
attr=np.stack([np.cos(phi),np.sin(phi),Vu[:,1],np.repeat(outer.astype(np.float32),3)],1)
dev='cuda'
glctx=dr.RasterizeCudaContext()
vt=torch.tensor(Vu,dtype=torch.float32,device=dev); ft=torch.tensor(Fu,dtype=torch.int32,device=dev)
at=torch.tensor(attr,dtype=torch.float32,device=dev)
nf=torch.tensor(np.repeat(fn,3,0),dtype=torch.float32,device=dev)
def render_attr(T, x0,y0,S,scale):
    Kc=K.copy(); Kc[0,2]=(K[0,2]-x0+0.5)*scale; Kc[1,2]=(K[1,2]-y0+0.5)*scale; Kc[:2,:2]*=scale
    res=int(S*scale)
    Tt=torch.tensor(T,dtype=torch.float32,device=dev)
    vc=vt@Tt[:3,:3].T+Tt[:3,3]
    fx,fy,cx,cy=Kc[0,0],Kc[1,1],Kc[0,2],Kc[1,2]; near,far=0.05,20
    x,yy,z=vc[:,0],vc[:,1],vc[:,2]
    clip=torch.stack([(2*fx*x+(2*cx-res)*z)/res,(-2*fy*yy-(2*cy-res)*z)/res,(far+near)/(far-near)*z-2*far*near/(far-near),z],-1)[None].contiguous()
    rast,_=dr.rasterize(glctx,clip,ft,resolution=[res,res])
    # view cosine
    nc=nf@Tt[:3,:3].T; vd=-vc/torch.linalg.norm(vc,dim=1,keepdim=True); cosv=(nc*vd).sum(1,keepdim=True)
    a,_=dr.interpolate(torch.cat([at,cosv],1)[None].contiguous(),rast,ft)
    m=(rast[...,3]>0)
    return a[0].flip(0).cpu().numpy(), m[0].flip(0).cpu().numpy()
NPHI=120; NH=12; HB=(-0.055,0.08)
def pano(i,T,per_mask=None,scale=4):
    """returns sum, weight arrays (NH,NPHI) of high-passed gray"""
    p=K@T[:3,3]; u,v=p[:2]/p[2]; S=int(K[0,0]*0.4/p[2])+8; x0=int(u-S/2); y0=int(v-S/2)
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    crop=np.zeros((S,S,3),np.uint8)
    ya,yb,xa,xb=max(y0,0),min(y0+S,1152),max(x0,0),min(x0+S,1536)
    crop[ya-y0:yb-y0,xa-x0:xb-x0]=img[ya:yb,xa:xb]
    g=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)
    g=cv2.resize(g,(S*scale,S*scale),interpolation=cv2.INTER_CUBIC)
    a,m=render_attr(T,x0,y0,S,scale)
    ok=m&(a[...,3]>0.99)&(a[...,4]>0.35)
    if per_mask is not None:
        pm=np.zeros((S,S),bool); pm[ya-y0:yb-y0,xa-x0:xb-x0]=per_mask[ya:yb,xa:xb]
        pm=cv2.dilate(pm.astype(np.uint8),np.ones((3,3),np.uint8))>0
        ok&=~cv2.resize(pm.astype(np.uint8),(S*scale,S*scale),interpolation=cv2.INTER_NEAREST).astype(bool)
    ph=np.degrees(np.arctan2(a[...,1],a[...,0]))%360; h=a[...,2]
    # high-pass: subtract local mean within object region
    w=ok.astype(np.float32); k=int(6*scale)|1
    mu=cv2.GaussianBlur(g*w,(k,k),0)/(cv2.GaussianBlur(w,(k,k),0)+1e-6)
    hp=g-mu
    pi=(ph[ok]/360*NPHI).astype(int)%NPHI; hi=np.clip(((h[ok]-HB[0])/(HB[1]-HB[0])*NH).astype(int),0,NH-1)
    sm=np.zeros((NH,NPHI)); wt=np.zeros((NH,NPHI))
    np.add.at(sm,(hi,pi),hp[ok]); np.add.at(wt,(hi,pi),1)
    return sm,wt

def pano_img(gimg, x0, y0, S, T, scale=4, permask=None):
    g=cv2.resize(gimg,(S*scale,S*scale),interpolation=cv2.INTER_CUBIC)
    a,m=render_attr(T,x0,y0,S,scale)
    ok=m&(a[...,3]>0.99)&(a[...,4]>0.35)
    ph=np.degrees(np.arctan2(a[...,1],a[...,0]))%360; h=a[...,2]
    w=ok.astype(np.float32); k=int(6*scale)|1
    mu=cv2.GaussianBlur(g*w,(k,k),0)/(cv2.GaussianBlur(w,(k,k),0)+1e-6)
    hp=g-mu
    pi=(ph[ok]/360*NPHI).astype(int)%NPHI; hi=np.clip(((h[ok]-HB[0])/(HB[1]-HB[0])*NH).astype(int),0,NH-1)
    sm=np.zeros((NH,NPHI)); wt=np.zeros((NH,NPHI))
    np.add.at(sm,(hi,pi),hp[ok]); np.add.at(wt,(hi,pi),1)
    return sm,wt

def texture_pano(dist=2.6, nview=12):
    R=Renderer(V,F,UV,TEX,K,1152,1536)
    from scipy.spatial.transform import Rotation as Rot
    sm=np.zeros((NH,NPHI)); wt=np.zeros((NH,NPHI))
    for k in range(nview):
        T=np.eye(4); T[:3,:3]=Rot.from_euler('x',-10,degrees=True).as_matrix()@Rot.from_euler('y',k*360/nview,degrees=True).as_matrix()
        T[:3,3]=[0,0,dist]; T[:3,3]-=T[:3,:3]@AXC
        p=K@T[:3,3]; u,v=p[:2]/p[2]; S=int(K[0,0]*0.4/p[2])+8; x0=int(u-S/2); y0=int(v-S/2)
        col,mk,_=R.render(T[None],crop=(x0,y0,S,S),scale=1)
        gimg=(col[0].cpu().numpy()@np.array([0.299,0.587,0.114])*255).astype(np.float32)
        s,w=pano_img(gimg,x0,y0,S,T); sm+=s; wt+=w
    return sm,wt

TXP=None
def corr_shift(sm,wt,sign=-1.0):
    """circular correlation of a frame panorama with the texture panorama; returns score per shift bin s,
    where s means: texture content at phi appears in the frame panorama at phi+s."""
    global TXP
    if TXP is None:
        d=np.load('/work/scratch/texpano.npz'); TXP=np.where(d['wt']>0,d['sm']/np.maximum(d['wt'],1),0)
    P=np.where(wt>0,sm/np.maximum(wt,1),0); W=(wt>0).astype(float)
    P=P-(P*W).sum(1,keepdims=True)/np.maximum(W.sum(1,keepdims=True),1); P*=W
    sc=np.zeros(NPHI)
    for s in range(NPHI):
        Ts=np.roll(TXP,s,axis=1)
        num=(P*Ts).sum(); den=np.sqrt((P**2).sum()*((Ts**2)*W).sum())+1e-9
        sc[s]=sign*num/den
    return sc
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    M[:3,3]=AXC-M[:3,:3]@AXC; return M
