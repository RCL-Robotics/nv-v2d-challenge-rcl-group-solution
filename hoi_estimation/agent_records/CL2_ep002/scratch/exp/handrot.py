import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
kp=rec['human_keypoints_world'].astype(np.float64)
def hand_frames(kp,base):
    h=kp[:,base:base+21]; wr=h[:,20]; mcp=h[:,[3,7,11,15,19]]
    x=mcp.mean(1)-wr; x/=np.linalg.norm(x,axis=1,keepdims=True)
    a=mcp[:,0]-mcp[:,4]; z=np.cross(x,a); z/=np.linalg.norm(z,axis=1,keepdims=True); y=np.cross(z,x)
    return np.stack([x,y,z],-1), wr
T=rec['T_world_mesh'].astype(np.float64) if sys.argv[1]=='base' else np.load(sys.argv[1])['T_world_mesh']
base=int(sys.argv[2]); a,b=int(sys.argv[3]),int(sys.argv[4])
Rh,wr=hand_frames(kp,base)
rel=np.einsum('fji,fjk->fik',Rh,T[:,:3,:3])
def yaw(R): x=R[:,:,0]; return np.degrees(np.arctan2(x[:,2],x[:,0]))
hy=np.degrees(np.arctan2(Rh[:,2,0],Rh[:,0,0]))
oy=yaw(T[:,:3,:3])
ref=rel[(a+b)//2]
for i in range(a,b,5):
    print(i, f'handyaw {hy[i]:7.1f} objyaw {oy[i]:7.1f} diff {((oy[i]-hy[i]+180)%360)-180:7.1f}  rel-vs-mid {rot_angle_deg(rel[i],ref):6.1f}')
