import numpy as np, json
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
kp=r['human_keypoints_world'].astype(np.float64)
z=Tc[:,2,3]; zs=savgol_filter(z,21,2)
handle=np.array([0,0,0.33])  # mesh-frame top
hw=(T[:,:3,:3]@handle)+T[:,:3,3]
lh=kp[:,42:63].mean(1); rh=kp[:,21:42].mean(1)
hc=(np.linalg.inv(Twc)[None,:3,:3]@lh[...,None])[...,0]+np.linalg.inv(Twc)[:3,3]
for i in range(150,640,6):
    print(i,'z %.3f hf %.3f'%(z[i],z[i]-zs[i]),'handle-lh',np.round(hw[i]-lh[i],3),'|%.3f|'%np.linalg.norm(hw[i]-lh[i]), 'lhz %.3f'%hc[i,2], 'cp',np.round(r['contact_prob_right_left'][i],2))
print('hf std', np.std((z-zs)[165:600]))
