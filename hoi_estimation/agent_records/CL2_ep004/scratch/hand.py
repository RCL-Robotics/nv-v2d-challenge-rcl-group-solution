import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from bat_lib import *
K,Twc,rec,plane=load_inputs('/work/inputs')
T=rec['T_world_mesh'].astype(np.float64); kp=rec['human_keypoints_world'].astype(np.float64); cp=rec['contact_prob_right_left']
# handle points in mesh frame: z from 0.1..0.3
hz=np.linspace(0.08,0.30,12); hp=np.stack([np.zeros(12),np.zeros(12),hz,np.ones(12)],1)
for i in range(0,803,20):
    H=(T[i]@hp.T)[:3].T
    rh=kp[i,21:42].mean(0); lh=kp[i,42:63].mean(0)
    dr=np.linalg.norm(H-rh,axis=1).min(); dl=np.linalg.norm(H-lh,axis=1).min()
    print(i,'cp',np.round(cp[i],2),'dR %.3f dL %.3f'%(dr,dl),'rh',np.round(rh,2),'lh',np.round(lh,2))
