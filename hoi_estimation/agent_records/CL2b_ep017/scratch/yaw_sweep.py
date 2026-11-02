import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
ep=Episode('/work/inputs')
rd=Renderer(ep.mesh)
c=np.array([0.0026,-0.003,0])
def about_axis(T,deg):
    A=np.eye(4);A[:3,3]=c; B=np.eye(4);B[:3,:3]=Rz(deg); C=np.eye(4);C[:3,3]=-c
    return T@A@B@C
frames=[int(a) for a in sys.argv[1].split(',')]
ths=np.arange(-45,46,3)
S=192
tot=np.zeros(len(ths))
for i in frames:
    T0=ep.Tc_base[i]
    ctr=proj(ep.K,T0[None,:3,3])[0]
    crop=crop_box(ctr,100)
    Ts=np.stack([about_axis(T0,t) for t in ths])
    m,_=rd.render(Ts,ep.K,crop,S)
    om=torch.tensor(cut(ep.obj_mask[i].astype(np.uint8)*255,crop,S,cv2.INTER_AREA)/255.,device='cuda').float()
    pm=torch.tensor(cut(ep.person_mask[i].astype(np.uint8)*255,crop,S,cv2.INTER_AREA)/255.,device='cuda').float()
    valid=1-pm*(1-om)  # ignore person pixels that are not object
    inter=(m*om*valid).sum((1,2)); uni=((m+om-m*om)*valid).sum((1,2))
    iou=(inter/uni).cpu().numpy(); tot+=iou
    print(i,' '.join('%d:%.3f'%(t,v) for t,v in zip(ths,iou)))
print('mean',' '.join('%d:%.4f'%(t,v) for t,v in zip(ths,tot/len(frames))))
