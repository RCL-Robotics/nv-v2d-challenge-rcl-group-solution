import cv2, numpy as np, h5py, sys
fr=np.load('/work/scratch/frames_half.npy',mmap_mode='r')
f=h5py.File('/work/inputs/masks.h5')
idx=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
tiles=[]
for i in idx:
    im=fr[i].copy()
    om=cv2.resize(f['object'][i].astype(np.uint8),(768,576))>0
    pm=cv2.resize(f['person'][i].astype(np.uint8),(768,576))>0
    im[om]=(0.5*im[om]+[0,0,127]).astype(np.uint8)
    im[pm]=(0.7*im[pm]+[76,0,0]).astype(np.uint8)
    im=cv2.resize(im,(384,288)); cv2.putText(im,str(i),(5,25),0,0.8,(0,255,0),2)
    tiles.append(im)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[k:k+6]) for k in range(0,len(tiles),6)]
cv2.imwrite(sys.argv[4],np.vstack(rows))
