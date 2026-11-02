import numpy as np
r=np.load('/work/inputs/reconstruction.npz'); kp=r['human_keypoints_world'].astype(np.float64)
i=300
print('elbow7',kp[i,7].round(3),'elbow8',kp[i,8].round(3))
for j in range(21,42): print(j, kp[i,j].round(3), 'd to elbow8 %.3f'%np.linalg.norm(kp[i,j]-kp[i,8]), 'd to elbow7 %.3f'%np.linalg.norm(kp[i,j]-kp[i,7]))
for j in range(62,70): print(j, kp[i,j].round(3), 'd to elbow8 %.3f'%np.linalg.norm(kp[i,j]-kp[i,8]))
