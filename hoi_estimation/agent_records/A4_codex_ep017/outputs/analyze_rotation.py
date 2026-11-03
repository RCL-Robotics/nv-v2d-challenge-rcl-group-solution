import numpy as np,json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
r=np.load('/work/inputs/reconstruction.npz');C=np.linalg.inv(json.load(open('/work/inputs/camera.json'))['T_world_cam']); T=C@r['T_world_mesh']; p=r['human_keypoints_world']@C[:3,:3].T+C[:3,3]
y=np.unwrap(np.arctan2(T[:,2,0],T[:,0,0]));plt.figure(figsize=(16,6));plt.plot(np.rad2deg(y),label='baseline yaw')
for a,b,name in [(5,6,'shoulders'),(7,8,'elbows'),(9,10,'wrists'),(42,21,'hand wrists'),(46,25,'index knuckles'),(50,29,'middle knuckles')]:
 d=p[:,a]-p[:,b]; hy=np.unwrap(np.arctan2(d[:,2],d[:,0]));hy+=np.round((y[125]-hy[125])/(2*np.pi))*2*np.pi;hy+=y[125]-hy[125];plt.plot(np.rad2deg(hy),label=name,alpha=.7)
plt.xlim(100,355);plt.grid();plt.legend();plt.savefig('/work/outputs/rotation_review.png')
print('coords',[(i,p[150,i].round(2).tolist()) for i in [5,6,7,8,9,10,21,42]])
