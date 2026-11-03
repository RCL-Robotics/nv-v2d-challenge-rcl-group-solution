import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, h5py, torch, cv2
from bat_lib import *
# usage: viz.py poses.npz key out.jpg frame1 frame2 ...
K,Twc,rec,plane=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
src,key,out=sys.argv[1:4]; frames=[int(x) for x in sys.argv[4:]]
T=np.load(src)[key].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@T
f=h5py.File('/work/inputs/masks.h5')
cap=cv2.VideoCapture('/work/inputs/video.mp4')
tiles=[]
# barrel marker: mesh points on axis
pts=np.array([[0,0,-0.2,1],[0,0,0.33,1]])
for i in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
    m,_=R.render(Tc[[i]],K,W_IMG,H_IMG); m=(m[0].cpu().numpy()>0).astype(np.uint8)
    om=f['object'][i].astype(np.uint8)
    ys,xs=np.nonzero(om|m)
    if len(xs)==0: cx,cy=768,576
    else: cx,cy=int(np.median(xs)),int(np.median(ys))
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,(0,255,0),1)
    cs,_=cv2.findContours(om,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,(255,0,255),1)
    p=(K@(Tc[i]@pts.T)[:3]).T; p=p[:,:2]/p[:,2:]
    cv2.circle(im,tuple(int(v) for v in p[0]),5,(0,255,255),-1)  # barrel end yellow
    cv2.circle(im,tuple(int(v) for v in p[1]),5,(255,255,0),-1)  # knob cyan
    x0=max(0,min(1536-300,cx-150)); y0=max(0,min(1152-300,cy-150))
    t=im[y0:y0+300,x0:x0+300].copy(); t=cv2.resize(t,(450,450),interpolation=cv2.INTER_NEAREST)
    cv2.putText(t,str(i),(5,30),0,1,(0,255,0),2); tiles.append(t)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+6]) for k in range(0,len(tiles),6)]))
