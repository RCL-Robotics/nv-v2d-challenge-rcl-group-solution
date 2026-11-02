import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from handframe import hand_frames
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); kp=r['human_keypoints_world'].astype(np.float64); T=r['T_world_mesh'].astype(np.float64)
H=hand_frames(kp); Rel=np.linalg.inv(H)@T
ref=Rel[300]
for i in range(170,625,8):
    d=np.degrees(np.linalg.norm(R.from_matrix(ref[:3,:3].T@Rel[i,:3,:3]).as_rotvec()))
    print(i,'rel rot vs f300 %.1f'%d,'rel t',np.round(Rel[i,:3,3],3), ' rotvec',np.round(np.degrees(R.from_matrix(ref[:3,:3].T@Rel[i,:3,:3]).as_rotvec()),0))
