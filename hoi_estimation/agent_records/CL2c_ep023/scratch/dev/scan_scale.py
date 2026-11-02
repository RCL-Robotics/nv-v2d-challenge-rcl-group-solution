import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from scipy.spatial import cKDTree
from solve import *
cam,K,Twc,rec,plane=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); mv=np.asarray(mesh.vertices)
Tcw=np.linalg.inv(Twc); Tb=rec['T_world_mesh'].astype(np.float64)
dg=np.load('/work/scratch/v1_diag.npz'); s_used=float(dg['scale'])
T1=np.load('/work/scratch/v1.npz')['T_world_mesh']; Tc1=Tcw@T1; Tc1[:,:3,3]/=s_used   # unscaled (h0 model)
zb=(Tcw@Tb)[:,2,3]; z0=Tc1[:,2,3]
nrm=plane[:3]/np.linalg.norm(plane[:3]); d=plane[3]/np.linalg.norm(plane[:3]); Dcam=nrm@Twc[:3,3]+d; h0=float(dg['h0'])
kp=rec['human_keypoints_world']; vs=mv[::10]; cp=rec['contact_prob_right_left'].max(1)
print('s_medz=%.4f  s_medratio=%.4f  s_static_only=%.4f'%(np.median(zb)/np.median(z0), np.median(zb/z0), np.median(np.r_[zb[:135],zb[878:]]/np.r_[z0[:135],z0[878:]])))
for s in [0.99,0.995,1.0,1.005,1.01,1.015,1.02,1.025,1.03]:
    T=np.array([np.vstack([np.hstack([Tc1[i,:3,:3],s*Tc1[i,:3,3:]]),[0,0,0,1]]) for i in range(946)]); T=Twc@T
    hd=[]
    for i in range(340,460,2):
        tr=cKDTree(vs@T[i,:3,:3].T+T[i,:3,3]); hd.append(tr.query(kp[i,21:63])[0].min())
    hd2=[]
    for i in range(135,878,6):
        tr=cKDTree(vs@T[i,:3,:3].T+T[i,:3,3]); hd2.append(tr.query(kp[i,21:63])[0].min())
    print('s %.3f support h %.1fcm | hand dist turn med %.1fcm | hand dist all-moving med %.1fcm | med dz %.1fcm'%(s,100*(Dcam-s*(Dcam-h0)),100*np.median(hd),100*np.median(hd2),100*np.median(s*z0-zb)))
