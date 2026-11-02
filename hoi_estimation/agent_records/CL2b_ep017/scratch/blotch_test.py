import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
import nvdiffrast.torch as dr
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
v=np.asarray(ep.mesh.vertices); az=np.degrees(np.arctan2(v[:,1]-AXIS_C[1],v[:,0]-AXIS_C[0]))
attr=np.stack([az, v[:,2]],1).astype(np.float32)
attr_t=torch.tensor(attr,device='cuda')
def render_attr(Tc,crop,size):
    x0,y0,s=crop; sc=size/s; K=ep.K
    fx,fy=K[0,0]*sc,K[1,1]*sc; cx,cy=(K[0,2]-x0)*sc,(K[1,2]-y0)*sc; n,fz=0.05,20.
    P=torch.tensor([[2*fx/size,0,2*cx/size-1,0],[0,2*fy/size,2*cy/size-1,0],[0,0,(fz+n)/(fz-n),-2*fz*n/(fz-n)],[0,0,1,0]],device='cuda',dtype=torch.float32)
    Tc=torch.tensor(Tc,device='cuda',dtype=torch.float32)
    vh=torch.cat([rd.v,torch.ones_like(rd.v[:,:1])],1)
    clip=torch.einsum('ij,bvj->bvi',P,torch.einsum('bij,vj->bvi',Tc,vh)).contiguous()
    rast,_=dr.rasterize(rd.ctx,clip,rd.f,(size,size))
    a,_=dr.interpolate(attr_t[None],rast,rd.f)
    return a.cpu().numpy(), (rast[...,3]>0).cpu().numpy()
a,b,T=int(sys.argv[1]),int(sys.argv[2]),np.load(sys.argv[3])
fr=ep.frames()[a:b].astype(np.float32)
ctr=proj(ep.K,(T@np.array([0,0,0.09,1]))[None,:3])[0]; h=int(0.2*ep.K[0,0]/T[2,3]); crop=crop_box(ctr,h)
x0,y0,s=crop
med=np.median(fr[:,y0:y0+s,x0:x0+s],0)/255.
S=s*2
med=cv2.resize(med,(S,S),interpolation=cv2.INTER_CUBIC)
mx=med.max(-1); mn=med.min(-1); vs=(mx-mn)/(mx+1e-3)
for off in [0,90,180,270]:
    att,cov=render_attr(axis_rot(T,off)[None],crop,S)
    azr=att[0,...,0]; zr=att[0,...,1]; cv0=cov[0]
    cv0=cv2.erode(cv0.astype(np.uint8),np.ones((5,5)))>0
    lower=cv0&(zr>-0.06)&(zr<0.02)
    bl=lower&(azr>95)&(azr<140)&(zr>-0.045)&(zr<0.01)
    rest=lower&~bl
    print('off',off,'blotch px',bl.sum(),'video sat blotch %.3f rest %.3f'%(vs[bl].mean() if bl.sum() else np.nan, vs[rest].mean()), 'value blotch %.3f rest %.3f'%(mx[bl].mean() if bl.sum() else np.nan, mx[rest].mean()))
cv2.imwrite(f'sat_{a}.png',(vs*255).astype(np.uint8))
f=lambda x: np.clip((x-0.7)/0.3,0,1)
tiles=[]
for off in [0,90,180,270]:
    att,cov=render_attr(axis_rot(T,off)[None],crop,S)
    azr=att[0,...,0]; zr=att[0,...,1]; cv0=cov[0]
    bl=cv0&(azr>95)&(azr<140)&(zr>-0.045)&(zr<0.01)
    im=np.repeat((f(vs)*255).astype(np.uint8)[...,None],3,2).copy()
    cs,_=cv2.findContours(bl.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im,cs,-1,(0,0,255),2)
    cs,_=cv2.findContours(cv0.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im,cs,-1,(0,255,0),1)
    tiles.append(cv2.resize(im,(300,300)))
cv2.imwrite(f'blotch_ov_{a}.png',np.hstack(tiles))
