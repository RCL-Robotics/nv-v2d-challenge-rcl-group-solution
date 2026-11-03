import sys
from joint import *
from traj import eval_ious
a,b=396,600
frames=np.arange(a,b)
train=frames[::2]; test=frames[1::2]
dtr=prep_frames(list(train)); dte=prep_frames(list(test))
Tinit=Tc[a:b].copy()
for lt,lr_ in [(0,0),(0.1,0.1),(1,1),(10,10),(100,100)]:
    t=time.time()
    T,iou,dl=joint_fit(Tinit, np.arange(b-a), list(train-a), dtr, iters=300, lam_t=lt, lam_r=lr_, verbose=False)
    ite=eval_ious(T[test-a],list(test),dte)
    dz=np.abs(np.diff(T[:,2,3])).mean()
    print(lt,lr_,'train %.4f test %.4f'%(iou.mean(),ite.mean()),'mean|dz| %.4f'%dz,'%.0fs'%(time.time()-t),flush=True)
    np.save(f'cvM_{lt}.npy',T)
