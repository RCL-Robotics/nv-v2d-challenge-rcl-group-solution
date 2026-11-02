import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
for (a,b,ref) in [(0,95,50),(345,440,400)]:
    fr=ep.frames()[a:b].astype(np.float32)
    # median to remove transient occluders
    T0=ep.Tc_base[ref]
    ctr=proj(ep.K,(T0@np.array([0,0,0.09,1]))[None,:3])[0]; h=int(0.22*ep.K[0,0]/T0[2,3]); crop=crop_box(ctr,h)
    x0,y0,s=crop
    med=np.median(fr[:,y0:y0+s,x0:x0+s],0)
    big=cv2.resize(med,(512,512),interpolation=cv2.INTER_CUBIC)
    Ts=np.stack([axis_rot(T0,o) for o in [0,90,180,270]])
    m,rgb=rd.render(Ts,ep.K,crop,512,color=True)
    rgb=rgb.cpu().numpy()*255
    cv2.imwrite(f'avg_{a}.jpg',np.hstack([big[:,:,::-1]]+[rgb[k][:,:,::-1] for k in range(4)]).astype(np.uint8))
