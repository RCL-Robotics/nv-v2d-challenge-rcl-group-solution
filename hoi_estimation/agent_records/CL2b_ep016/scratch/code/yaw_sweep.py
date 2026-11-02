import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
frames=[int(x) for x in sys.argv[1].split(',')]
angs=np.arange(0,360,15)
tot=np.zeros((len(angs),2))
for i in frames:
    box=frame_box(K,Tc[i]); hw=(160,160)
    img,om,pm=fd.crop(i,box,hw)
    cands=[]
    for a in angs:
        X=Tc[i].copy(); X[:3,:3]=Tc[i,:3,:3]@rot_about_axis([0,0,1],np.radians(a)); cands.append(X)
    s=score_batch(Rd,K,box,hw,np.array(cands),img,om,pm)
    tot+=np.c_[s['iou'],s['col']]
    print(i,' '.join('%3d:%.2f/%.2f'%(a,x,y) for a,x,y in zip(angs,s['iou'],s['col'])))
print('mean',' '.join('%3d:%.2f/%.2f'%(a,x,y) for a,(x,y) in zip(angs,tot/len(frames))))
