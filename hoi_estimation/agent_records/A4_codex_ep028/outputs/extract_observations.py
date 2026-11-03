import cv2,h5py,numpy as np,json,time
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
p=Path('/work/inputs');cap=cv2.VideoCapture(str(p/'video.mp4'));f=h5py.File(p/'masks.h5');F=f['object'].shape[0]; rec=[];t=time.time()
for i in range(F):
 ok,im=cap.read();mask=f['object'][i].astype('uint8');yy,xx=np.where(mask);x0=max(0,xx.min()-15);x1=min(im.shape[1],xx.max()+16);y0=max(0,yy.min()-15);y1=min(im.shape[0],yy.max()+16);roi=im[y0:y1,x0:x1];ms=mask[y0:y1,x0:x1];h=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV);o=((h[:,:,0]<24)&(h[:,:,1]>130)&(h[:,:,2]>60)&cv2.dilate(ms,np.ones((5,5),np.uint8)).astype(bool)).astype('uint8');n,lab,stats,cent=cv2.connectedComponentsWithStats(o);oranges=[]
 for k in range(1,n):
  if stats[k,4]>=4:oranges.append([*list(cent[k]+[x0,y0]),int(stats[k,4])])
 pts=np.c_[xx,yy].astype(float);cen=pts.mean(0);_,_,vh=np.linalg.svd((pts[::max(1,len(pts)//3000)]-cen),full_matrices=False);axis=vh[0];axis=axis if axis[1]>0 else -axis;proj=(pts-cen)@axis;side=(pts-cen)@np.array([-axis[1],axis[0]]);lo,hi=np.percentile(proj,[0.1,99.9]);bins=np.linspace(lo,hi,31);width=[];count=[];mid=[]
 for l,r in zip(bins[:-1],bins[1:]):
  ss=side[(proj>=l)&(proj<r)];width.append(float(np.percentile(ss,95)-np.percentile(ss,5)) if len(ss)>3 else 0);count.append(len(ss));mid.append(float(np.median(ss)) if len(ss) else 0)
 rec.append(dict(frame=i,orange=oranges,center=cen.tolist(),axis=axis.tolist(),bounds=[lo,hi],width=width,count=count,mid=mid,area=len(xx),bbox=[int(x0),int(y0),int(x1),int(y1)]))
 if i%200==0:print(i,round(time.time()-t,1),flush=True)
json.dump(rec,open('/work/outputs/observations.json','w'));print('done',time.time()-t)
