from evalm import *
import cv2
r=np.load('/work/inputs/reconstruction.npz')
Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
for i in [0,60,100,830,850]:
    c=cent[i]; x0=int(c[0]-64); y0=int(c[1]-64)
    _,m,_=R.render(Tb[i],K,128,128,x0,y0); m=m[0].cpu().numpy()
    o=OM[i,y0:y0+128,x0:x0+128]
    ys,xs=np.nonzero(m); yo,xo=np.nonzero(o)
    print(i,'render area',m.sum(),'mask',o.sum(),'render bbox',xs.min(),xs.max(),ys.min(),ys.max(),'mask bbox',xo.min(),xo.max(),yo.min(),yo.max())
    img=cv2.imread(f'frames/{i:04d}.jpg')[y0:y0+128,x0:x0+128]
    cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(img,cnt,-1,(0,0,255),1)
    cnt,_=cv2.findContours(o.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(img,cnt,-1,(0,255,0),1)
    cv2.imwrite(f'cont_{i}.png',cv2.resize(img,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST))
