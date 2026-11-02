import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
from solve import evidence, detect_phases
cam,K,Twc,rec=load_inputs('/work/inputs'); Tcw=np.linalg.inv(Twc)
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
e1,s2,area,_=detect_phases(fd.om)
Tb=Tcw[None]@rec['T_world_mesh'].astype(float)
odd=[i for i in range(e1+1,s2) if i%2==1 and area[i]>=400]
for p in ['base']+sys.argv[1:]:
    Tc=Tb if p=='base' else Tcw[None]@np.load(p)['T_world_mesh']
    r=[]
    for i in odd:
        box=frame_box(K,Tb[i],scale=1.3); img,om,pm=fd.crop(i,box,(192,192))
        t,u=evidence(Rd,K,box,(192,192),Tc[i:i+1],img,om,pm); r.append((t[0],u[0]))
    r=np.array(r); seg=lambda a,b: r[[k for k,i in enumerate(odd) if a<=i<b]].mean(0)
    print(p,'heldout tex %.3f iou %.3f'%tuple(r.mean(0)),' | 123-250 %.3f/%.3f 250-320 %.3f/%.3f 320-376 %.3f/%.3f'%(*seg(123,250),*seg(250,320),*seg(320,376)))
