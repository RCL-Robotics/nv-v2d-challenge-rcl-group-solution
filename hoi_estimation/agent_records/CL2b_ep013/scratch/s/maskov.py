import cv2, numpy as np, h5py, sys
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
idx=[int(x) for x in sys.argv[1].split(',')]; x0,y0,w,h=[int(x) for x in sys.argv[2].split(',')]
tiles=[]
for t in idx:
    im=np.ascontiguousarray(fr[t,y0:y0+h,x0:x0+w]).copy()
    for key,col in [('object',(0,255,0)),('person',(255,0,0))]:
        m=f[key][t][y0:y0+h,x0:x0+w].astype(np.uint8)
        cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cnt,-1,col,2)
    cv2.putText(im,str(t),(5,25),0,0.8,(0,255,255),2); tiles.append(im)
nc=int(sys.argv[4])
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[k:k+nc]) for k in range(0,len(tiles),nc)]))
