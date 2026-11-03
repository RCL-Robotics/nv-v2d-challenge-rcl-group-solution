import numpy as np, cv2, h5py, sys
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
bg=np.load('/work/scratch/bg.npy').astype(np.float32)
def newmask(t):
    im=fr[t].astype(np.float32)
    d=np.abs(im-bg).max(-1)
    fg=d>40
    hsv=cv2.cvtColor(fr[t],cv2.COLOR_BGR2HSV)
    bright=(hsv[...,2]>110)
    m=fg&bright
    m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3)))
    return m>0
idx=[int(x) for x in sys.argv[1].split(',')]; x0,y0,w,h=[int(x) for x in sys.argv[2].split(',')]
tiles=[]
for t in idx:
    im=np.ascontiguousarray(fr[t,y0:y0+h,x0:x0+w]).copy()
    m=newmask(t)[y0:y0+h,x0:x0+w]
    im[m]=(0.5*im[m]+[0,0,127]).astype(np.uint8)
    for key,col in [('object',(0,255,0)),('person',(255,0,0))]:
        mm=f[key][t][y0:y0+h,x0:x0+w].astype(np.uint8)
        cnt,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cnt,-1,col,1)
    cv2.putText(im,str(t),(5,25),0,0.8,(0,255,255),2); tiles.append(im)
nc=int(sys.argv[4])
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[k:k+nc]) for k in range(0,len(tiles),nc)]))
