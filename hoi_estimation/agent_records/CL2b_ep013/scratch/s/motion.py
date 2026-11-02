import numpy as np,h5py,cv2
fr=np.load('/work/scratch/frames_half.npy',mmap_mode='r')
f=h5py.File('/work/inputs/masks.h5')
F=922; d=np.full(F,np.nan)
om=np.stack([cv2.resize(f['object'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST) for i in range(F)])>0
pm=np.stack([cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST) for i in range(F)])>0
np.save('/work/scratch/om_half.npy',om); np.save('/work/scratch/pm_half.npy',pm)
k=3
for t in range(k,F):
    m=(om[t]|om[t-k]) & ~pm[t] & ~pm[t-k]
    m=cv2.dilate(m.astype(np.uint8),np.ones((5,5)))>0
    m&=~(cv2.dilate((pm[t]|pm[t-k]).astype(np.uint8),np.ones((7,7)))>0)
    if m.sum()<50: continue
    d[t]=np.abs(fr[t].astype(float)-fr[t-k].astype(float)).mean(-1)[m].mean()
np.save('/work/scratch/motion.npy',d)
for t in range(0,F,5): print(t, f'{d[t]:.2f}', end=' | ' if t%50!=45 else '\n')
