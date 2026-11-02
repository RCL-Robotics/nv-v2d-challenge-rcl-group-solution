import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py
from hoi_lib import *
v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
K,Twc,Kp,W,H=load_camera('/work/inputs')
r=np.load('/work/inputs/reconstruction.npz')
Twm=r['T_world_mesh'].astype(np.float64)
Tcm=np.linalg.inv(Twc)@Twm
R=Renderer(v,f,uv,tex)
x0,y0,w,h=640,520,240,180
frames=[int(a) for a in sys.argv[1:]] or [0,100,200,300,400,500]
om=h5py.File('/work/inputs/masks.h5','r')['object']
tiles=[]
for i in frames:
    rgb,m,d=R.render(Tcm[i],K,x0,y0,w,h)
    rgb=(rgb[0].cpu().numpy()*255).astype(np.uint8)[...,::-1]; m=m[0].cpu().numpy()
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+h,x0:x0+w]
    ov=im.copy(); ov[m]=rgb[m]
    mk=om[i][y0:y0+h,x0:x0+w].astype(np.uint8)
    cs,_=cv2.findContours(mk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,cs,-1,(0,255,0),1)
    t=np.hstack([im,ov]); t=cv2.resize(t,None,fx=2,fy=2); cv2.putText(t,str(i),(3,20),0,0.7,(0,255,255),2); tiles.append(t)
cv2.imwrite('/work/scratch/render_test.jpg', np.vstack([np.hstack(tiles[j:j+2]) for j in range(0,len(tiles),2)]))
