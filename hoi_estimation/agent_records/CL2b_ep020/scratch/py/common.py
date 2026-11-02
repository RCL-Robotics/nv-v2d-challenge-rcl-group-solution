import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py
from hoi_lib import *
v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
K,Twc,Kp,W,H=load_camera('/work/inputs')
Tcw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz')
Twm=r['T_world_mesh'].astype(np.float64)
Tcm=Tcw@Twm
hf=h5py.File('/work/inputs/masks.h5','r'); OM=hf['object']; PM=hf['person']
up_cam=Tcw[:3,:3]@np.array([0,-1.,0])
center=(v.min(0)+v.max(0))/2
Rd=Renderer(v,f,uv,tex)
def frame(i): return cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
import json, torch
nup=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]); nup/=np.linalg.norm(nup)
def upright_basis(R0):
    # minimal rotation taking R0 @ (-e_z) onto nup
    a=R0@np.array([0,0,-1.]); ax=np.cross(a,nup); s=np.linalg.norm(ax); c=a@nup
    Rc=rotvec_to_R(ax/s*np.arctan2(s,c)) if s>1e-9 else np.eye(3)
    return Rc@R0
B=upright_basis(Twm[0,:3,:3])
def yaw_of(Rwm):
    M=B.T@Rwm  # ~ Rz(theta)
    return np.arctan2(M[...,1,0],M[...,0,0])
yaw_base=np.unwrap(yaw_of(Twm[:,:3,:3]))
Rcw=Tcw[:3,:3]; tcw=Tcw[:3,3]
