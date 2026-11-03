import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
import nvdiffrast.torch as dr
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
v=np.asarray(ep.mesh.vertices); az=np.degrees(np.arctan2(v[:,1]-AXIS_C[1],v[:,0]-AXIS_C[0]))
attr_t=torch.tensor(np.stack([az,v[:,2]],1).astype(np.float32),device='cuda')
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
d=np.load('carry_profile.npz'); fr=list(d['frames']); g=d['grid']; iou=d['iou'].reshape(len(fr),len(g)); Tf=d['Tf'].reshape(len(fr),len(g),4,4)
R0=ep.T_base[0,:3,:3]
S=160
rows=[]
for k,i in enumerate(fr):
    if i%2: continue
    j=iou[k].argmax(); T=Tf[k,j]
    ctr=proj(ep.K,(T@np.array([0,0,0.09,1]))[None,:3])[0]; h=int(0.2*ep.K[0,0]/T[2,3]); crop=crop_box(ctr,h)
    im=cut(ep.frames()[i],crop,S,cv2.INTER_AREA).astype(np.float32)/255
    mx=im.max(-1); mn=im.min(-1); vs=(mx-mn)/(mx+1e-3)
    om=cut(ep.obj_mask[i].astype(np.uint8),crop,S,cv2.INTER_NEAREST)>0
    pm=cut(ep.person_mask[i].astype(np.uint8),crop,S,cv2.INTER_NEAREST)>0
    om=cv2.erode(om.astype(np.uint8),np.ones((3,3)))>0
    tw=twist_swing(R0.T@(ep.T_world_cam@T)[:3,:3])[0]
    Ts=np.stack([axis_rot(T,o) for o in [0,90,180,270]])
    att,cov=render_attr(Ts,crop,S)
    res=[]
    for b in range(4):
        azr,zr=att[b,...,0],att[b,...,1]; c=cov[b]&om&~pm
        lower=c&(zr>-0.06)&(zr<0.03)
        bl=lower&(azr>100)&(azr<135)&(zr>-0.04)&(zr<0.005)
        ctl=lower&~((azr>80)&(azr<155))
        res.append((bl.sum(), vs[bl].mean()-vs[ctl].mean() if bl.sum()>=6 else np.nan))
    rows.append((i,tw,res))
    print(i,'tw %5.0f'%tw,' | '.join('off%3d n%4d c% .3f'%(o,n,c) for o,(n,c) in zip([0,90,180,270],res)))
