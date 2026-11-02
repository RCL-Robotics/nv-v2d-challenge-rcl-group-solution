from common import *
from viz import show
# args: out frame deg1 deg2 ...
out=sys.argv[1]; i=int(sys.argv[2]); degs=[float(a) for a in sys.argv[3:]]
L=[]
for d in degs:
    T=Tcm.copy(); T[i]=yaw_pose(Tcm[i],up_cam,center,np.radians(d)); L.append(T)
show(L,[i],out)
