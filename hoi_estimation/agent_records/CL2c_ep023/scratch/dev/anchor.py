import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp); mesh=load_mesh(inp)
fm=FloorModel(mesh,plane,np.array([-0.28,-0.02,-0.96]),h0=0.014)
Tcw=np.linalg.inv(Twc)
Tb=Tcw@rec['T_world_mesh'].astype(np.float64); To=Tcw@np.load(sys.argv[1])['T_world_mesh']
c=fm.c
def zc(T): return np.array([(t[:3,:3]@c+t[:3,3])[2] for t in T])
zb,zo=zc(Tb),zc(To)
phases={'static0':(0,96),'push_away':(96,340),'turn_occl':(340,460),'return_backforth':(460,893),'static1':(893,946),'all':(0,946)}
for k,(a,b) in phases.items(): print('%-17s base med z %.3f ours %.3f  diff mean %.1f med %.1f cm   ratio %.4f'%(k,np.median(zb[a:b]),np.median(zo[a:b]),100*np.mean(zo[a:b]-zb[a:b]),100*np.median(zo[a:b]-zb[a:b]),np.median(zb[a:b])/np.median(zo[a:b])))
s=np.median(zb)/np.median(zo); print('global scale',s)
# support height after scaling about camera centre: plane distance from camera
o=Twc[:3,3]; n,d=fm.n,fm.d
Dcam=n@o+d   # camera height above plane
# support plane: n.x + d = h0 ; scaled about o by s: new height = Dcam - s*(Dcam-h0)
print('camera height above plane %.3f'%Dcam, 'support height after scaling %.3f'%(Dcam-s*(Dcam-0.014)))
