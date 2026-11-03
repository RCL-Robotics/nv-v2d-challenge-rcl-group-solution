import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
frames=[int(x) for x in sys.argv[1].split(',')]; yaws=[float(x) for x in sys.argv[2].split(',')]
rows=[]
for i in frames:
    B=Tc[i]; box=frame_box(K,B,scale=1.2); hw=(200,200)
    img=fd.crop(i,box,hw)[0]; row=[img]
    for a in yaws:
        X=B.copy(); X[:3,:3]=B[:3,:3]@rot_about_axis([0,0,1],np.radians(a))
        rgb,m,_=Rd.render(torch.tensor(X[None],device='cuda'),K,box,hw)
        r=img.copy(); mk=m[0].cpu().numpy(); r[mk]=rgb[0].cpu().numpy()[mk]
        r=cv2.GaussianBlur(r,(0,0),1.0*hw[0]/(box[2]-box[0])); row.append(r)
    o=(np.hstack(row)[...,::-1]*255).astype(np.uint8).copy(); cv2.putText(o,str(i),(3,18),0,0.6,(0,0,255),2); rows.append(o)
cv2.imwrite(sys.argv[3],np.vstack(rows))
