import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
frames=[int(x) for x in sys.argv[1].split(',')]
rows=[]
for i in frames:
    X0=Tc[i]
    head=project(K,X0[:3,:3]@[0,0,0.12]+X0[:3,3]); s=0.12*K[0,0]/X0[2,3]
    box=(head[0]-s/2,head[1]-s/2,head[0]+s/2,head[1]+s/2); hw=(240,240)
    img=fd.crop(i,box,hw)[0]
    row=[img]
    for a in [0,90,180,270]:
        X=X0.copy(); X[:3,:3]=X0[:3,:3]@rot_about_axis([0,0,1],np.radians(a))
        rgb,m,_=Rd.render(torch.tensor(X[None],device='cuda'),K,box,hw)
        r=img*0.5; mk=m[0].cpu().numpy(); r[mk]=rgb[0].cpu().numpy()[mk]; row.append(r)
    o=(np.hstack(row)[...,::-1]*255).astype(np.uint8).copy(); cv2.putText(o,str(i),(3,18),0,0.6,(0,255,0),2); rows.append(o)
cv2.imwrite(sys.argv[2],np.vstack(rows))
