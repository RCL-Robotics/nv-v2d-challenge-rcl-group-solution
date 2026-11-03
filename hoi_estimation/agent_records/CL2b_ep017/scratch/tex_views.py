import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
tiles=[]
for i,zoom in [(0,1),(400,1)]:
    T0=ep.Tc_base[i]
    ctr=proj(ep.K,T0[None,:3,3])[0]; crop=crop_box(ctr,int(70*1.92/T0[2,3]*1.0)+10)
    Ts=np.stack([axis_rot(T0,a) for a in [0,90,180,270]])
    m,rgb=rd.render(Ts,ep.K,crop,384,color=True)
    rgb=rgb.cpu().numpy()
    im=cut(ep.frames()[i],crop,384).astype(np.float32)/255
    row=[im]+[rgb[k]+(1-m[k].cpu().numpy()[...,None])*im*0.3 for k in range(4)]
    tiles.append(np.hstack(row))
cv2.imwrite('texviews.jpg',(np.vstack(tiles)[:,:,::-1]*255).clip(0,255).astype(np.uint8))
