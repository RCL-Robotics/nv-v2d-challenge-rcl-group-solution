import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, time
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
fp=rec['T_cam_mesh_foundationpose'].astype(float)
a0,a1,step=[int(x) for x in sys.argv[1:4]]
names=['base','fp','y90','y180','y270','x180','y180x180']
res={}
t=time.time()
for i in range(a0,a1,step):
    B=Tc[i]; H=[]
    H.append(B)
    X=B.copy(); X[:3,:3]=fp[i,:3,:3]; H.append(X)
    for a in [90,180,270]:
        X=B.copy(); X[:3,:3]=B[:3,:3]@rot_about_axis([0,0,1],np.radians(a)); H.append(X)
    X=B.copy(); X[:3,:3]=B[:3,:3]@rot_about_axis([1,0,0],np.pi); H.append(X)
    X=B.copy(); X[:3,:3]=B[:3,:3]@rot_about_axis([1,0,0],np.pi)@rot_about_axis([0,0,1],np.pi); H.append(X)
    H=np.array(H)
    box=frame_box(K,B); hw=(192,192)
    img,om,pm=fd.crop(i,box,hw)
    if om.sum()<50: continue
    Tn,s=refine_batch(Rd,K,box,hw,H,img,om,pm,iters=60,w_col=0.5)
    res[i]=dict(T=Tn,**s)
    k=np.argmin(s['loss'])
    print(i,'best',names[k],' '.join('%s:%.3f(%.2f,%.2f)'%(n,l,d,c) for n,l,d,c in zip(names,s['loss'],s['dice'],s['col'])),'| base moved %.1f, best-vs-base %.0f'%(rot_angle_deg(B[:3,:3],Tn[0,:3,:3]),rot_angle_deg(B[:3,:3],Tn[k,:3,:3])),flush=True)
print('time',time.time()-t)
np.save(f'/work/scratch/carry_hyp_{a0}_{a1}_{step}.npy',res,allow_pickle=True)
