import sys; sys.path.insert(0,'/work/outputs')
from lib import *
exec(open('/work/scratch/code/upright.py').read())
phases=[(0,133),(210,273),(345,408),(474,531),(600,642),(687,780)]
# crops are centred per frame on mask bbox; for static use a fixed window from phase median offset
fr_all=read_video('/work/inputs/video.mp4')
f=h5py.File('/work/inputs/masks.h5')
med=[]; offs=[]
for a,b in phases:
    off=np.median(D['off'][a:b],0).astype(int); x0,y0=off
    stack=[];vm=[]
    for i in range(a,b):
        im=fr_all[i][y0:y0+160,x0:x0+160].astype(np.float32)
        pm=f['person'][i][y0:y0+160,x0:x0+160]; om=f['object'][i][y0:y0+160,x0:x0+160]
        ok=~(pm&~om)
        stack.append(np.where(ok[...,None],im,np.nan))
    md=np.nanmedian(np.stack(stack),0); med.append(md); offs.append(off)
np.savez('static_median.npz',med=np.stack(med),off=np.stack(offs),phases=np.array(phases))
t=[]
for k,md in enumerate(med):
    c=np.nan_to_num(md[35:125,40:120]).astype(np.uint8); c=cv2.resize(c,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    cv2.putText(c,str(phases[k]),(5,25),0,0.7,(0,0,255),2); t.append(c)
cv2.imwrite('medians.jpg',np.hstack(t))
