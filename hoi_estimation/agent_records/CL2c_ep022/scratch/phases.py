import numpy as np, cv2, h5py
f=h5py.File('/work/inputs/masks.h5')
F=719
mo=f['object'][:,::2,::2]; mp=f['person'][:,::2,::2]
imgs=np.stack([cv2.cvtColor(cv2.imread('frames/%04d.jpg'%(i+1)),cv2.COLOR_BGR2GRAY)[::2,::2] for i in range(F)]).astype(np.float32)
cent=np.array([np.argwhere(m).mean(0) if m.any() else [np.nan,np.nan] for m in mo])
diff=np.zeros(F); 
for i in range(1,F):
    j=max(0,i-5)
    reg=(mo[i]|mo[j]); reg=cv2.dilate(reg.astype(np.uint8),np.ones((5,5)))>0
    reg&=~cv2.dilate((mp[i]|mp[j]).astype(np.uint8),np.ones((9,9))).astype(bool)
    d=np.abs(cv2.GaussianBlur(imgs[i],(5,5),0)-cv2.GaussianBlur(imgs[j],(5,5),0))
    diff[i]=(d[reg]>20).mean() if reg.sum()>50 else np.nan
np.savez('phase_stats.npz',cent=cent,diff=diff)
for i in range(0,F,5): print(i, 'cent %.1f %.1f'%tuple(cent[i]), 'diff %.3f'%diff[i], 'occl %.2f'%((mp[i]).sum()/(1+mo[i].sum())))
