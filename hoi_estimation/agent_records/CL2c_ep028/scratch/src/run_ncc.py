import numpy as np,cv2,time,sys
from photo2 import *
d=np.load(sys.argv[1]); R,t=d['R'],d['t']
F,G=d['L'].shape
cap=cv2.VideoCapture('/work/inputs/video.mp4')
N=np.full((F,G),np.nan); t0=time.time()
for i in range(F):
    ok,im=cap.read()
    Ts=[Rt_to_T(R[i,h],t[i,h]) for h in range(G)]
    N[i]=orange_ncc(i,Ts,im)
np.save(sys.argv[2],N); print(time.time()-t0)
