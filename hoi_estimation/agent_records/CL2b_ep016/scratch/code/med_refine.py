import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, time
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
a0,a1,ref=[int(x) for x in sys.argv[1:4]]
fr=range(a0,a1)
box=frame_box(K,Tc[ref],scale=1.4); hw=(256,256)
cr=[fd.crop(i,box,hw) for i in fr]
med=np.median(np.stack([c[0] for c in cr]),0).astype(np.float32)
om=np.mean(np.stack([c[1] for c in cr]),0)>0.5
pm=np.mean(np.stack([c[2] for c in cr]),0)>0.5
R0=rot_mean(Tc[list(fr),:3,:3]); t0=np.median(Tc[list(fr),:3,3],0)
angs=np.arange(0,360,30)
X=np.repeat(np.eye(4)[None],len(angs),0)
for k,a in enumerate(angs): X[k,:3,:3]=R0@rot_about_axis([0,0,1],np.radians(a)); X[k,:3,3]=t0
t=time.time()
for wc in [0.0,0.5,1.0]:
    Tn,s=refine_batch(Rd,K,box,hw,X,med,om,pm,iters=150,w_col=wc)
    print('w_col',wc,time.time()-t)
    for k,a in enumerate(angs):
        dyaw=np.degrees(np.arctan2(*(R0.T@Tn[k,:3,:3])[[1,0],0]))
        print('init %3d -> loss %.3f dice %.3f col %.3f | dR from init %.1f, yaw %.0f, tilt %.1f, t-t0 %s'%(a,s['loss'][k],s['dice'][k],s['col'][k],rot_angle_deg(X[k,:3,:3],Tn[k,:3,:3]),dyaw,np.degrees(np.arccos(np.clip((R0.T@Tn[k,:3,:3])[2,2],-1,1))),np.round((Tn[k,:3,3]-t0)*100,1)))
