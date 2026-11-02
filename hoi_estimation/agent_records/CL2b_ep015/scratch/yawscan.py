import sys; sys.path.insert(0,'/work/scratch'); sys.path.insert(0,'/work/outputs')
from static_test import *
import time
cxy=np.array([0.005,-0.004,0])
def Rz(phi):
    T=np.eye(4); c,s=np.cos(phi),np.sin(phi); T[:2,:2]=[[c,-s],[s,c]]; T[:3,3]=cxy-T[:3,:3]@cxy; return T
yaws=np.radians(np.arange(0,360,10))
idx=list(range(0,715,4))
t0=time.time()
IOU=np.zeros((len(idx),len(yaws))); PH=np.zeros_like(IOU); LS=np.zeros_like(IOU)
for k0 in range(0,len(idx),30):
    sub=idx[k0:k0+30]
    for j,yw in enumerate(yaws):
        Tc=Tc0@Rz(yw)
        _,iou,ls,lp=refine_fz(sub,Tc,iters=1,lam=0.3)
        IOU[k0:k0+len(sub),j]=iou; PH[k0:k0+len(sub),j]=lp; LS[k0:k0+len(sub),j]=ls
print(time.time()-t0)
np.savez('yawscan.npz',IOU=IOU,PH=PH,LS=LS,idx=idx,yaws=yaws)
