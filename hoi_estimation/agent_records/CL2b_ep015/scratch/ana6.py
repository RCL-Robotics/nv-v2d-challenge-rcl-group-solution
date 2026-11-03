import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64)
kp=r['human_keypoints_world'].astype(np.float64)
def ang(v): return np.degrees(np.arctan2(v[...,2],v[...,0]))
ys=ang(T[:,:3,0])   # mesh x axis yaw
lat=(kp[:,5]-kp[:,6])+(kp[:,9]-kp[:,10]); yb=ang(lat)
rel=(ys-yb+180)%360-180
hand=kp[:,21:42].mean(1); 
for i in range(150,640,6):
    print(i,'sign %.0f body %.0f rel %.0f'%(ys[i],yb[i],rel[i]), 'hand-pelvis lateral offset %.2f'%np.dot(hand[i]-(kp[i,9]+kp[i,10])/2, lat[i]/np.linalg.norm(lat[i])))
