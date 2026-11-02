import numpy as np, cv2, h5py, sys
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
ref=np.load('/work/scratch/sam_ref.npy',allow_pickle=True).item()
idx=[int(x) for x in sys.argv[1].split(',')]; tiles=[]
for t in idx:
    im=np.ascontiguousarray(fr[t]).copy(); om=f['object'][t]
    m=np.unpackbits(ref[t])[:1152*1536].reshape(1152,1536).astype(bool) if t in ref else om
    ys,xs=np.nonzero(om|m); cx,cy=int(np.median(xs)),int(np.median(ys))
    im[m]=(0.5*im[m]+[0,0,127]).astype(np.uint8)
    for mm,col in [(om,(0,255,0)),(f['person'][t],(255,0,0))]:
        cnt,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cnt,-1,col,1)
    crop=np.pad(im,((200,200),(200,200),(0,0)))[cy:cy+400,cx:cx+400]
    crop=cv2.resize(crop,(300,300)); cv2.putText(crop,str(t),(5,25),0,0.7,(0,255,255),2); tiles.append(crop)
nc=6
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2],np.vstack([np.hstack(tiles[k:k+nc]) for k in range(0,len(tiles),nc)]))
