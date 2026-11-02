import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
a0,a1,ref=[int(x) for x in sys.argv[1:4]]
fr=range(a0,a1)
box=frame_box(K,Tc[ref],scale=1.4); hw=(320,320)
cr=[fd.crop(i,box,hw) for i in fr]
med=np.median(np.stack([c[0] for c in cr]),0).astype(np.float32)
om=np.mean(np.stack([c[1] for c in cr]),0)>0.5
pm=np.mean(np.stack([c[2] for c in cr]),0)>0.5
angs=np.arange(0,360,10)
X=np.repeat(Tc[ref][None],len(angs),0)
for k,a in enumerate(angs): X[k,:3,:3]=Tc[ref,:3,:3]@rot_about_axis([0,0,1],np.radians(a))
s=score_batch(Rd,K,box,hw,X,med,om,pm)
print(' '.join('%d:%.2f/%.2f'%(a,x,y) for a,x,y in zip(angs,s['iou'],s['col'])))
show=[int(x) for x in sys.argv[4].split(',')]
row=[med]
mm=(om*255).astype(np.uint8); cnt,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for a in show:
    Y=Tc[ref].copy(); Y[:3,:3]=Tc[ref,:3,:3]@rot_about_axis([0,0,1],np.radians(a))
    rgb,m,_=Rd.render(torch.tensor(Y[None],device='cuda'),K,box,hw)
    r=med.copy(); mk=m[0].cpu().numpy(); r[mk]=rgb[0].cpu().numpy()[mk]; row.append(r)
o=(np.hstack(row)[...,::-1]*255).astype(np.uint8).copy()
for k in range(len(row)): cv2.drawContours(o[:,k*320:(k+1)*320],cnt,-1,(0,0,255),1)
cv2.imwrite(sys.argv[5],o)
