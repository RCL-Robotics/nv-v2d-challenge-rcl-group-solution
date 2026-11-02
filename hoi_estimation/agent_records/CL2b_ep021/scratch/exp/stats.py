import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs')
Tw=rec['T_world_mesh'].astype(np.float64); F=len(Tw)
Tc=cam_poses_from_world(Twc,Tw)
dR=rot_angles_deg(Tw[:-1,:3,:3],Tw[1:,:3,:3]); dt=np.linalg.norm(Tw[1:,:3,3]-Tw[:-1,:3,3],axis=1)
cp=rec['contact_prob_right_left']
# mesh z axis in world (stool up axis)
up=Tw[:,:3,2]
n=np.array(gp['plane'][:3])
for i in range(0,F,10):
    print(i, 'pos',np.round(Tw[i,:3,3],3),'z',round(Tc[i,2,3],3),'dR%.1f dt%.3f'%(dR[min(i,F-2)],dt[min(i,F-2)]), 'up·n %.3f'%(up[i]@n), 'cp',np.round(cp[i],2))
print('big jumps', [(i,round(dR[i],1)) for i in np.nonzero(dR>5)[0]])
