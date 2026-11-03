import cv2, numpy as np, sys
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r')
a,b,s=[int(x) for x in sys.argv[1:4]]; x0,y0,w,h=[int(x) for x in sys.argv[4].split(',')]
tiles=[]
for t in range(a,b+1,s):
    im=cv2.resize(np.ascontiguousarray(fr[t,y0:y0+h,x0:x0+w]),(int(w*float(sys.argv[6])),int(h*float(sys.argv[6]))))
    cv2.putText(im,str(t),(5,25),0,0.8,(0,255,255),2); tiles.append(im)
nc=int(sys.argv[7]) if len(sys.argv)>7 else 6
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[5],np.vstack([np.hstack(tiles[k:k+nc]) for k in range(0,len(tiles),nc)]))
