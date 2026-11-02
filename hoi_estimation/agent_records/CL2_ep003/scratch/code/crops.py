import cv2,numpy as np,h5py,sys
idx=[int(x) for x in sys.argv[2].split(',')]
out=sys.argv[1]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
f=h5py.File('/work/inputs/masks.h5')
tiles=[]
i=0; want=set(idx)
while True:
    ok,im=cap.read()
    if not ok or i>max(idx): break
    if i in want:
        m=f['object'][i]
        ys,xs=np.nonzero(m)
        if len(ys): cy,cx=int(ys.mean()),int(xs.mean())
        else: cy,cx=500,800
        h=80
        c=im[max(cy-h,0):cy+h, max(cx-h,0):cx+h].copy()
        mm=m[max(cy-h,0):cy+h, max(cx-h,0):cx+h]
        cnt,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        c2=c.copy(); cv2.drawContours(c2,cnt,-1,(0,255,0),1)
        c=np.hstack([c,c2]); c=cv2.resize(c,(c.shape[1]*2,c.shape[0]*2))
        cv2.putText(c,str(i),(5,25),0,0.8,(0,0,255),2)
        tiles.append(cv2.resize(c,(640,320)))
    i+=1
rows=[np.hstack(tiles[k:k+3]) if len(tiles[k:k+3])==3 else np.hstack(tiles[k:k+3]+[np.zeros_like(tiles[0])]*(3-len(tiles[k:k+3]))) for k in range(0,len(tiles),3)]
cv2.imwrite(out,np.vstack(rows))
