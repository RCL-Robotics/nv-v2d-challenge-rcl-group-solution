import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
a0,a1,ref=[int(x) for x in sys.argv[1:4]]; yaws=[float(x) for x in sys.argv[4].split(',')]
fr=list(range(a0,a1))
R0=rot_mean(Tc[fr,:3,:3]); t0=np.median(Tc[fr,:3,3],0)
c=project(K,t0); s=0.36*K[0,0]/t0[2]
box=(c[0]-s/2,c[1]-s/2,c[0]+s/2,c[1]+s/2); hw=(360,360)
med=np.median(np.stack([fd.crop(i,box,hw)[0] for i in fr]),0)
def enh(x): 
    lo,hi=np.percentile(x,1),np.percentile(x,99); return np.clip((x-lo)/(hi-lo),0,1)
row=[enh(med)]
for a in yaws:
    X=np.eye(4); X[:3,:3]=R0@rot_about_axis([0,0,1],np.radians(a)); X[:3,3]=t0
    rgb,m,_=Rd.render(torch.tensor(X[None],device='cuda'),K,box,hw)
    r=med.copy(); mk=m[0].cpu().numpy(); r[mk]=rgb[0].cpu().numpy()[mk]
    r=cv2.GaussianBlur(r,(0,0),1.5*hw[0]/s)
    row.append(enh(r))
o=(np.hstack(row)[...,::-1]*255).astype(np.uint8).copy()
for k,a in enumerate(['med']+yaws): cv2.putText(o,str(a),(k*360+5,20),0,0.6,(0,0,255),2)
cv2.imwrite(sys.argv[5],o)
