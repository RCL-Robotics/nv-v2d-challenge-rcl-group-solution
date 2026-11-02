import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py
from hoi_lib import *
a,b,st=map(int,sys.argv[1:4]); out=sys.argv[4]
idx=list(range(a,b,st))
fr=read_video('/work/inputs/video.mp4',idx)
mf=h5py.File('/work/inputs/masks.h5')
tiles=[]
for i,f in zip(idx,fr):
    om=mf['object'][i]; ys,xs=np.nonzero(om)
    cx,cy=(int(xs.mean()),int(ys.mean())) if len(xs) else (768,700)
    s=170; x0=np.clip(cx-s,0,1536-2*s); y0=np.clip(cy-s,0,1152-2*s)
    t=f[y0:y0+2*s,x0:x0+2*s].copy()
    cs,_=cv2.findContours(om[y0:y0+2*s,x0:x0+2*s].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    t=np.ascontiguousarray(t[:,:,::-1]); cv2.drawContours(t,cs,-1,(0,255,0),1)
    cv2.putText(t,str(i),(5,25),0,0.8,(0,255,255),2); tiles.append(t)
while len(tiles)%5: tiles.append(np.zeros_like(tiles[0]))
rows=[np.concatenate(tiles[k:k+5],1) for k in range(0,len(tiles),5)]
cv2.imwrite(out,np.concatenate(rows,0))
