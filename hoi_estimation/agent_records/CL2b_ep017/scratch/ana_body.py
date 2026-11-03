import numpy as np
r=np.load('/work/inputs/reconstruction.npz')
K=r['human_keypoints_world'].astype(np.float64); T=r['T_world_mesh'].astype(np.float64)
def yaw(v): # world y-down; horizontal plane x-z; angle about -y (up)
    return np.degrees(np.arctan2(v[...,0],v[...,2]))
sh=K[:,5]-K[:,6]; hp=K[:,9]-K[:,10]
up=np.array([0,-1.,0])
face=np.cross(up, sh+hp)   # direction
hw=K[:,62]-K[:,41]
def handyaw(w,i,p):  # hand plane x-axis: index MCP - pinky MCP
    return yaw(K[:,i]-K[:,p])
ry=handyaw(None,28,40); ly=handyaw(None,49,61)
obj=T[:,:3,3]
for i in range(100,345,3):
    dR=np.linalg.norm(K[i,41]-obj[i]); dL=np.linalg.norm(K[i,62]-obj[i])
    print(i,'face %.0f'%yaw(face[i]),'LRwrist %.0f'%yaw(hw[i]),'rh %.0f lh %.0f'%(ry[i],ly[i]),'dR %.2f dL %.2f'%(dR,dL), 'cp',np.round(r['contact_prob_right_left'][i],2))
