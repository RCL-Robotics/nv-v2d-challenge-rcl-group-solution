import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from common import read_video
from scipy.ndimage import median_filter
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']
fr=read_video('/work/inputs/video.mp4',scale=0.5)
g=np.stack([cv2.GaussianBlur(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),(5,5),0) for f in fr]).astype(np.float32)
np.save('/work/scratch/gray_half.npy',g.astype(np.uint8))
F=len(g); k=3; ker=np.ones((7,7),np.uint8)
s=np.zeros(F)
for i in range(F-k):
    reg=cv2.dilate((obj[i]|obj[i+k]).astype(np.uint8),ker)>0
    pv=cv2.dilate((per[i]|per[i+k]).astype(np.uint8),ker)>0
    r=reg&~pv
    s[i]=np.abs(g[i]-g[i+k])[r].mean() if r.sum()>50 else np.nan
s[F-k:]=s[F-k-1]
np.save('/work/scratch/photo_s.npy',s)
print(np.round(s,1).tolist())
