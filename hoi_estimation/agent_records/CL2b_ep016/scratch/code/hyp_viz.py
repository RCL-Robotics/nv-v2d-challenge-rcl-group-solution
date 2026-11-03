import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
res=np.load(sys.argv[1],allow_pickle=True).item()
frames=[int(x) for x in sys.argv[2].split(',')]; hy=[int(x) for x in sys.argv[3].split(',')]
rows=[]
for i in frames:
    box=frame_box(K,Tc[i]); hw=(192,192)
    img,om,pm=fd.crop(i,box,hw)
    row=[img]
    for X in [Tc[i]]+[res[i]['T'][h] for h in hy]:
        rgb,m,_=Rd.render(torch.tensor(X[None],device='cuda'),K,box,hw)
        r=img*0.4; mk=m[0].cpu().numpy(); r[mk]=rgb[0].cpu().numpy()[mk]; row.append(r)
    o=(np.hstack(row)[...,::-1]*255).astype(np.uint8).copy(); cv2.putText(o,str(i),(3,18),0,0.6,(0,255,0),2)
    rows.append(o)
cv2.imwrite(sys.argv[4],np.vstack(rows))
