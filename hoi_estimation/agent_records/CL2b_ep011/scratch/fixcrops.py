import cv2,numpy as np,sys
fr=[int(a) for a in sys.argv[2].split(',')]; cx,cy=int(sys.argv[3]),int(sys.argv[4]); s=int(sys.argv[5]) if len(sys.argv)>5 else 60
cap=cv2.VideoCapture('/work/inputs/video.mp4'); out=[]; i=0
while i<=max(fr):
    ok,f=cap.read()
    if i in fr:
        c=cv2.resize(f[cy-s*3//4:cy+s*3//4,cx-s:cx+s],(400,300),interpolation=cv2.INTER_CUBIC); cv2.putText(c,str(i),(5,25),0,0.8,(0,0,255),2); out.append(c)
    i+=1
while len(out)%5: out.append(np.zeros_like(out[0]))
cv2.imwrite(sys.argv[1],np.vstack([np.hstack(out[r*5:r*5+5]) for r in range(len(out)//5)]))
