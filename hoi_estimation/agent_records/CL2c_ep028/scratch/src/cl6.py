import numpy as np,cv2,h5py
from opt import *
from rend import Renderer
from core import rot_angle_deg
d=np.load('../run2/diag.npz'); c=np.load('../run1/stage1.npz')
h5=h5py.File('/work/inputs/masks.h5')
SS=4; Kh=K.copy(); Kh[:2]*=SS; Rn=Renderer(Kh,W=W*SS,H=H*SS)
rows=[]
for i in [30,90]:
    Rs=d['Rf'][i]; ts=d['tf'][i]/d['scale'][i]
    ang=rot_angle_deg(c['Rc'][i],Rb[i]); L=c['Lc'][i].copy(); L[ang>30]=9; g=L.argmin()
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg'); om=h5['object'][i]
    cs,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    outs=[]
    for k,(R,t) in enumerate([(Rs,ts),(c['Rc'][i,g],c['tc'][i,g])]):
        T=Rt_to_T(R,t)
        p=T[:3,:3]@np.array([0,0,-0.47])+T[:3,3]; u=K@p; u=(u[:2]/u[2]).astype(int)
        if k==0: x0,y0=u[0]-40,u[1]-40
        col,m,_=Rn.render(T,crop=(x0*SS,y0*SS,80*SS,80*SS)); mm=m[0].cpu().numpy().astype(np.uint8)
        a=cv2.resize(im[y0:y0+80,x0:x0+80],(320,320),interpolation=cv2.INTER_CUBIC)
        cc=[(cn-[x0,y0])*4 for cn in cs]; cv2.drawContours(a,cc,-1,(0,0,255),1)
        rc,_=cv2.findContours(cv2.resize(mm,(320,320),interpolation=cv2.INTER_NEAREST),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(a,rc,-1,(0,255,0),1)
        cv2.putText(a,f'{i} {"static(flip)" if k==0 else "base-roll"}',(4,16),0,0.5,(0,255,255),1); outs.append(a)
    rows.append(np.hstack(outs))
cv2.imwrite('../viz/cl6.jpg',np.vstack(rows))
