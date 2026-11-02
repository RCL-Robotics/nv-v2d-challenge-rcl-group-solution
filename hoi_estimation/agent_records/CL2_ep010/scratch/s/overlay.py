import sys; sys.path.insert(0,'/work/scratch/s')
from rend import *
import cv2, h5py
r=Renderer(); K,Kp,Twc=load_cam()
T=np.load(sys.argv[1])['T_world_mesh'].astype(np.float64) if len(sys.argv)>1 else np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
out=sys.argv[2] if len(sys.argv)>2 else 'ov.jpg'
frames=[int(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else list(range(0,794,50))
Tc=np.linalg.inv(Twc)@T
mk=h5py.File('/work/inputs/masks.h5')['object']
cap=cv2.VideoCapture('/work/inputs/video.mp4'); ims=[]
for i in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=cap.read()
    m,d,c=r.render(Tc[i],K,1152,1536)
    m=m[0].cpu().numpy(); om=mk[i]
    o=f.copy()
    o[m]=(0.5*o[m]+0.5*(c[0].cpu().numpy()[m][:,::-1]*255)).astype(np.uint8)
    cnt,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(o,cnt,-1,(0,0,255),2)
    cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(o,cnt,-1,(0,255,0),2)
    ys,xs=np.where(om|m); 
    if len(ys)==0: ys,xs=np.array([500,700]),np.array([600,900])
    cy,cx=int(ys.mean()),int(xs.mean()); y0=max(0,cy-250); x0=max(0,cx-330)
    crop=o[y0:y0+500,x0:x0+660]; crop=cv2.resize(crop,(660,500))
    iou=(m&om).sum()/max(1,(m|om).sum())
    cv2.putText(crop,f'{i} iou{iou:.2f}',(10,30),0,1,(0,0,255),2); ims.append(crop)
while len(ims)%4: ims.append(np.zeros_like(ims[0]))
cv2.imwrite(out,np.vstack([np.hstack(ims[k:k+4]) for k in range(0,len(ims),4)]))
