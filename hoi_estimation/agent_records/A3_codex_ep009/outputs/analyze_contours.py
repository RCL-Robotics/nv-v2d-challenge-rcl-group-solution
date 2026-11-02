import json,h5py,cv2,numpy as np,time
from contour_fit import prepare,fit_contour
from PIL import Image,ImageDraw
cv2.setNumThreads(1)
p='/work/inputs/';d=dict(np.load('/work/outputs/diagnostics/alignment.npz'));orig=d['origins'];side=d['render'].shape[-1]
shifts=[];scores=[];coverage=[];prepared=[]
t0=time.perf_counter()
with h5py.File(p+'masks.h5') as f:
 for i in range(len(orig)):
  x,y=orig[i];obs=d['object'][i];person=f['person'][i,y:y+side,x:x+side];pp=prepare(obs,person,d['rgb'][i]);prepared.append(pp)
  v,s=fit_contour(d['render'][i],pp);shifts.append(v);scores.append(s);coverage.append((d['render'][i]*pp[1]).sum()/max(d['render'][i].sum(),1))
shifts=np.array(shifts);scores=np.array(scores);coverage=np.array(coverage)
np.savez_compressed('/work/outputs/diagnostics/contour_fit.npz',shifts=shifts,scores=scores,coverage=coverage)
for a,b in [(0,150),(150,220),(220,490),(490,620),(620,790),(790,870)]:
 print(a,b,'shift median',np.median(shifts[a:b],axis=0),'mean score',scores[a:b].mean(0),'coverage',coverage[a:b].mean(),flush=True)
frames=[0,100,170,200,230,260,300,325,350,400,450,500,550,600,630,650,680,710,740,770,820,869]
w=224
sheet=Image.new('RGB',(6*w,4*w),'white');draw=ImageDraw.Draw(sheet)
for j,i in enumerate(frames):
 im=d['rgb'][i].copy();m=d['render'][i];mm=cv2.warpAffine(m,np.array([[1,0,shifts[i,0]],[0,1,shifts[i,1]]],np.float32),(side,side))
 for ma,col in [(m,(0,255,255)),(mm,(255,0,255))]:
  cs,_=cv2.findContours((ma>.5).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,col,1)
 sheet.paste(Image.fromarray(cv2.resize(im,(w,w))),(j%6*w,j//6*w));draw.text((j%6*w+3,j//6*w+3),f'{i} dxdy {shifts[i].round(1)} cost {scores[i].round(1)}',fill='white',stroke_fill='black',stroke_width=1)
sheet.save('/work/outputs/diagnostics/contour_translation.jpg')
print('elapsed',time.perf_counter()-t0)
