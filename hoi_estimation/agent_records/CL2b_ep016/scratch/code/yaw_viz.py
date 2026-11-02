import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
i=int(sys.argv[1]); angs=[int(a) for a in sys.argv[2].split(',')]
ref=int(sys.argv[3]) if len(sys.argv)>3 else i
box=frame_box(K,Tc[ref],scale=1.2); hw=(240,240)
img,om,pm=fd.crop(i,box,hw)
row=[img]
for a in angs:
    X=Tc[ref].copy(); X[:3,:3]=Tc[ref,:3,:3]@rot_about_axis([0,0,1],np.radians(a))
    rgb,m,_=Rd.render(torch.tensor(X[None],device='cuda'),K,box,hw)
    r=rgb[0].cpu().numpy(); r[~m[0].cpu().numpy()]=0.3; row.append(r)
out=(np.hstack(row)[...,::-1]*255).astype(np.uint8)
for k,a in enumerate(['img']+angs): cv2.putText(out,str(a),(k*240+5,20),0,0.6,(0,255,0),2)
cv2.imwrite(sys.argv[4] if len(sys.argv)>4 else '/work/scratch/viz/yaw.jpg',out)
