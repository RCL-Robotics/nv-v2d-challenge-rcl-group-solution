import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
ep=Episode('/work/inputs')
fl=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
tiles=[]
for i in fl:
    m=ep.obj_mask[i]; ys,xs=np.nonzero(m)
    T0=ep.Tc_base[i]
    if len(xs)<200: ctr=proj(ep.K,(T0@np.array([0,0,0.09,1]))[None,:3])[0]
    else: ctr=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])
    h=int(0.24*ep.K[0,0]/T0[2,3]); crop=crop_box(ctr,h)
    c=cut(ep.frames()[i],crop,200,cv2.INTER_CUBIC)[:,:,::-1].copy()
    cv2.putText(c,str(i),(3,18),0,0.6,(0,255,0),2); tiles.append(c)
nc=int(sys.argv[5]) if len(sys.argv)>5 else 8
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[4],np.vstack([np.hstack(tiles[k:k+nc]) for k in range(0,len(tiles),nc)]))
