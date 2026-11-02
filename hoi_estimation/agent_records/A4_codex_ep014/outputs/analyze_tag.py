import numpy as np,cv2,json,h5py
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
p='/work/inputs/';K=np.array(json.load(open(p+'camera.json'))['K']);WC=np.array(json.load(open(p+'camera.json'))['T_world_cam']);T=np.linalg.inv(WC)@np.load(p+'reconstruction.npz')['T_world_mesh'];tag=np.array([-.16595,-.15357,-.08807])
def project(t,pts):
 q=pts@t[:3,:3].T+t[:3,3];return q[:,:2]/q[:,2:]*K.diagonal()[:2]+K[:2,2]
ids=list(range(500,851,25));c=cv2.VideoCapture(p+'video.mp4');out=Image.new('RGB',(5*320,3*340),'white');draw=ImageDraw.Draw(out)
with h5py.File(p+'masks.h5') as h:
 for j,i in enumerate(ids):
  c.set(cv2.CAP_PROP_POS_FRAMES,i);_,bgr=c.read();im=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);angle=np.linspace(0,2*np.pi,100);ring=np.zeros(im.shape[:2],np.uint8)
  for rad,col in [(.31,1),(.12,0)]:
   pts=np.c_[rad*np.cos(angle),rad*np.sin(angle),np.ones(len(angle))*-.089];xy=project(T[i],pts);cv2.fillPoly(ring,[xy.astype(np.int32)],col)
  a=im.astype(float);sel=(a.min(2)>85)&(a.min(2)>.66*a.max(2))&(ring>0)&h['object'][i]
  n,lab,stats,centers=cv2.connectedComponentsWithStats(sel.astype(np.uint8))
  candidates=[]
  for k in range(1,n):
   area=stats[k,4]
   if 6<=area<=200:
    candidates.append((int(area),centers[k]));cv2.circle(im,tuple(centers[k].astype(int)),6,(255,0,0),1)
  # Fit a continuous tag angle for each candidate, print nearest discrete phase.
  angles=np.arange(360);rots=Rotation.from_euler('z',angles,degrees=True).as_matrix();pred=project(T[i],np.einsum('fij,j->fi',rots,tag))
  print(i,'normal',T[i,:3,2],[(area,np.round(xy,1).tolist(),int(np.argmin(np.linalg.norm(pred-xy,axis=-1))),float(np.linalg.norm(pred-xy,axis=-1).min())) for area,xy in candidates])
  for phase in [0,72,144,216,288]:
   xy=pred[phase];cv2.circle(im,tuple(xy.astype(int)),4,(0,255,0),1);cv2.putText(im,str(phase),tuple(xy.astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,255,0),1)
  m=h['object'][i];ys,xs=np.where(m);side=int(max(np.ptp(xs),np.ptp(ys))*1.05);cx=(xs.min()+xs.max())//2;cy=(ys.min()+ys.max())//2;x=cx-side//2;y=cy-side//2
  out.paste(Image.fromarray(im[y:y+side,x:x+side]).resize((320,320)),(j%5*320,j//5*340));draw.text((j%5*320,j//5*340+320),str(i),fill='black')
out.save('/work/scratch/tag.jpg')
