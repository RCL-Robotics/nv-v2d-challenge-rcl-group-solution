import cv2,numpy as np,h5py,sys
f=h5py.File('/work/inputs/masks.h5'); om=f['object']; pm=f['person']
frames=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
cap=cv2.VideoCapture('/work/inputs/video.mp4')
out=[]; i=0; last=(640,700)
want=set(frames)
while i<=max(frames):
    ok,fr=cap.read()
    if i in want:
        m=om[i]; ys,xs=np.nonzero(m)
        if len(xs): last=(int(xs.mean()),int(ys.mean()))
        cx,cy=last; s=100
        x0=max(cx-s,0); y0=max(cy-s,0)
        c=fr[y0:y0+2*s, x0:x0+2*s].copy()
        mm=m[y0:y0+2*s, x0:x0+2*s]
        cnt,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        c2=cv2.resize(c,(400,400),interpolation=cv2.INTER_CUBIC)
        cnt=[x*2 for x in cnt]
        cv2.drawContours(c2,cnt,-1,(0,255,0),1)
        cv2.putText(c2,str(i),(5,25),0,0.8,(0,0,255),2)
        out.append(c2)
    i+=1
while len(out)%6: out.append(np.zeros_like(out[0]))
g=np.vstack([np.hstack(out[r*6:(r+1)*6]) for r in range(len(out)//6)])
cv2.imwrite(sys.argv[4],g)
