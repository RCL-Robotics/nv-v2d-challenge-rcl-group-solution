import numpy as np, time
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r')
t0=time.time(); idx=np.arange(0,922,4)
bg=np.zeros((1152,1536,3),np.uint8)
for y in range(0,1152,96):
    bg[y:y+96]=np.median(fr[idx,y:y+96],axis=0).astype(np.uint8)
np.save('/work/scratch/bg.npy',bg); print(time.time()-t0)
import cv2; cv2.imwrite('/work/scratch/bg.jpg',cv2.resize(bg,(768,576)))
