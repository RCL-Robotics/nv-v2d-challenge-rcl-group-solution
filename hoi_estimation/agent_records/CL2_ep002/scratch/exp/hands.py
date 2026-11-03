import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs')
kp=rec['human_keypoints_world']; T=rec['T_world_mesh']
c=T[:,:3,3]+np.einsum('fij,j->fi',T[:,:3,:3],np.array([0.004,0.01,0.003]))
rh=kp[:,21:42].mean(1); lh=kp[:,42:63].mean(1)
cp=rec['contact_prob_right_left']
fig,ax=plt.subplots(3,1,figsize=(18,10),sharex=True)
ax[0].plot(np.linalg.norm(rh-c,axis=1),label='R-obj'); ax[0].plot(np.linalg.norm(lh-c,axis=1),label='L-obj'); ax[0].set_ylim(0,1)
ax[1].plot(cp[:,0],label='cR'); ax[1].plot(cp[:,1],label='cL')
for k,l in [(0,'x'),(1,'y'),(2,'z')]:
    ax[2].plot(rh[:,k],label='rh'+l); ax[2].plot(lh[:,k],'--',label='lh'+l)
for a in ax: a.grid(); a.legend()
ax[0].set_xticks(range(0,980,20)); plt.tight_layout(); plt.savefig('/work/scratch/vis/hands.png',dpi=60)
