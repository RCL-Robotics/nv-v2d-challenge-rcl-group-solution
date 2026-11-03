import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
fp=r['T_cam_mesh_foundationpose'].astype(np.float64); cc=r['T_cam_mesh_coconet'].astype(np.float64)
kp=r['human_keypoints_world'].astype(np.float64)
def yaw(Rm): # angle of mesh x-axis in world x-z plane
    a=Rm[...,:3,0]; return np.degrees(np.arctan2(a[...,2],a[...,0]))
def yawc(Rm): a=Rm[...,:3,0]; return np.degrees(np.arctan2(a[...,2],a[...,0]))
yb=yaw(T); yf=yawc(fp); yc=yawc(cc); ybc=yawc(Tc)
# body yaw from keypoints: try indices for hips (MHR70: 9,10 = left/right hip?) - print a few kp to guess
print(kp[0,:15].round(2))
for i in range(0,715,8): print(i,'world yaw base %.0f | cam yaw base %.0f fp %.0f cc %.0f'%(yb[i],ybc[i],yf[i],yc[i]))
