import sys,time; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py
from hoi_lib import *
t=time.time()
inp='/work/inputs'; mf=h5py.File(inp+'/masks.h5')
ids=list(range(0,946,6))
fr=np.stack(read_video(inp+'/video.mp4',ids))
k=np.ones((31,31),np.uint8)
bad=np.stack([cv2.dilate((mf['object'][i]|mf['person'][i]).astype(np.uint8),k)>0 for i in ids])
print('loaded',time.time()-t)
bg=np.zeros(fr.shape[1:],np.uint8); cnt=(~bad).sum(0)
for y0 in range(0,1152,64):
    a=fr[:,y0:y0+64].astype(np.float32); a[bad[:,y0:y0+64]]=np.nan
    bg[y0:y0+64]=np.nan_to_num(np.nanmedian(a,0)).astype(np.uint8)
print('min count',cnt.min(), (cnt<5).sum(), time.time()-t)
np.save('/work/scratch/bg.npy',bg); cv2.imwrite('/work/scratch/frames/bg.jpg',bg[:,:,::-1])
