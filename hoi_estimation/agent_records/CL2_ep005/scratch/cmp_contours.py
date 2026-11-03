import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def getframe(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=cap.read(); return f
def draw(i, Tcs, cols, out=None, pad=60):
    im=getframe(i); im=np.clip(im.astype(np.float32)*2.2,0,255).astype(np.uint8)
    allm=None
    for Tc,col in zip(Tcs,cols):
        r=R.render(Tc[None],d['K'],d['H'],d['W']); m=r['mask'][0].cpu().numpy().astype(np.uint8)
        # depth edges (internal box edges) via depth discontinuity + silhouette
        cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cnt,-1,col,1)
        allm=m if allm is None else allm|m
    ys,xs=np.where(allm); y0,y1,x0,x1=max(ys.min()-pad,0),ys.max()+pad,max(xs.min()-pad,0),xs.max()+pad
    crop=im[y0:y1,x0:x1]; crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    if out: cv2.imwrite(out,crop)
    return crop
import torch, nvdiffrast.torch as dr
fn=torch.tensor(np.asarray(mesh.face_normals),dtype=torch.float32,device='cuda')
def normal_edges(Tc):
    T=torch.tensor(Tc[None],dtype=torch.float32,device='cuda')
    K=d['K']; H,W=d['H'],d['W']
    P=torch.tensor([[2*K[0,0]/W,0,2*(K[0,2]+0.5)/W-1,0],[0,2*K[1,1]/H,2*(K[1,2]+0.5)/H-1,0],[0,0,1.002,-0.1],[0,0,1,0]],dtype=torch.float32,device='cuda')
    vh=torch.cat([R.v,torch.ones_like(R.v[:,:1])],1); vc=torch.einsum('bij,nj->bni',T,vh); clip=torch.einsum('ij,bnj->bni',P,vc).contiguous()
    rast,_=dr.rasterize(R.glctx,clip,R.f,(H,W))
    tri=rast[0,...,3].long()-1; nm=torch.zeros(H,W,3,device='cuda'); ok=tri>=0
    nm[ok]=fn[tri[ok]]@T[0,:3,:3].T
    nm=nm.cpu().numpy(); 
    # quantize to dominant box axes -> label image
    Rm=Tc[:3,:3]; lab=np.zeros((H,W),np.int32)
    dots=np.einsum('hwc,ck->hwk',nm,np.concatenate([Rm,-Rm],1))
    lab=np.argmax(dots,2)+1; lab[~ok.cpu().numpy()]=0
    e=np.zeros((H,W),np.uint8)
    e[:,1:]|=(lab[:,1:]!=lab[:,:-1]); e[1:,:]|=(lab[1:,:]!=lab[:-1,:])
    return e
def draw_edges(i,Tcs,cols,out,pad=60):
    im=getframe(i); im=np.clip(im.astype(np.float32)*2.2,0,255).astype(np.uint8); allm=None
    for Tc,col in zip(Tcs,cols):
        e=normal_edges(Tc); im[e>0]=col; allm=e if allm is None else allm|e
    ys,xs=np.where(allm); y0,y1,x0,x1=max(ys.min()-pad,0),ys.max()+pad,max(xs.min()-pad,0),xs.max()+pad
    crop=cv2.resize(im[y0:y1,x0:x1],None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST); cv2.imwrite(out,crop)
