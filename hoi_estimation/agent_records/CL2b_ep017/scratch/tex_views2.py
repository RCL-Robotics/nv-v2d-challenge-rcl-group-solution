import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
fl=[int(a) for a in sys.argv[1].split(',')]
tiles=[]
for i in fl:
    T0=ep.Tc_base[i]
    ctr=proj(ep.K,(T0@np.array([0,0,0.09,1]))[None,:3])[0]; crop=crop_box(ctr,int(0.25*ep.K[0,0]/T0[2,3]))
    Ts=np.stack([axis_rot(T0,a) for a in [0,90,180,270]])
    m,rgb=rd.render(Ts,ep.K,crop,256,color=True)
    rgb=rgb.cpu().numpy()
    im=cut(ep.frames()[i],crop,256,cv2.INTER_CUBIC).astype(np.float32)/255
    row=[im]+[rgb[k]+(1-m[k].cpu().numpy()[...,None])*im*0.3 for k in range(4)]
    r=np.hstack(row); cv2.putText(r,str(i),(5,20),0,0.6,(0,1,0),2); tiles.append(r)
cv2.imwrite(sys.argv[2],(np.vstack(tiles)[:,:,::-1]*255).clip(0,255).astype(np.uint8))
