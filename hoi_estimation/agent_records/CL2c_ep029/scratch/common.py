import numpy as np, json, h5py, trimesh
from scipy.spatial.transform import Rotation as R
INP='/work/inputs'
def load():
    r=np.load(f'{INP}/reconstruction.npz')
    cam=json.load(open(f'{INP}/camera.json'))
    K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Kp=np.array(cam['K_pipeline'])
    return r,K,Twc,Kp
def rotdeg(Ra,Rb):
    Rr=Ra.transpose(0,2,1)@Rb if Ra.ndim==3 else Ra.T@Rb
    tr=np.trace(Rr,axis1=-2,axis2=-1)
    return np.degrees(np.arccos(np.clip((tr-1)/2,-1,1)))
