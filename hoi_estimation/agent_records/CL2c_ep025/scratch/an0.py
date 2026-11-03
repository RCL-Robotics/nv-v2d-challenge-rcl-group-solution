import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz')
cam=json.load(open('/work/inputs/camera.json'))
Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
T=r['T_world_mesh'].astype(np.float64)
Tc=Tcw[None]@T
z=T[:,:3,2]  # bottle axis in world
np.set_printoptions(precision=3,suppress=True,linewidth=200)
cp=r['contact_prob_right_left']
for i in range(0,859,10):
    print(i, Tc[i,:3,3], 'axisW',z[i], 'cp',cp[i])
