from common import *
x0,y0,w,h=700,560,110,95
prev=None; res=[]
ref0=frame(0)[y0:y0+h,x0:x0+w].astype(np.float32)
refE=frame(552)[y0:y0+h,x0:x0+w].astype(np.float32)
for i in range(553):
    im=cv2.GaussianBlur(frame(i)[y0:y0+h,x0:x0+w].astype(np.float32),(5,5),1)
    pm=cv2.dilate(PM[i][y0:y0+h,x0:x0+w].astype(np.uint8),np.ones((9,9),np.uint8))>0
    om=OM[i][y0:y0+h,x0:x0+w]
    val=om&~pm
    d1=np.abs(im-prev).mean(-1)[val].mean() if prev is not None else 0
    d0=np.abs(im-cv2.GaussianBlur(ref0,(5,5),1)).mean(-1)[val].mean()
    dE=np.abs(im-cv2.GaussianBlur(refE,(5,5),1)).mean(-1)[val].mean()
    res.append((d1,d0,dE,val.sum(),om.sum()-val.sum()))
    prev=im
res=np.array(res); np.save('/work/scratch/motion.npy',res)
for i in range(0,553,4): print(i,' '.join('%.1f'%x for x in res[i]))
