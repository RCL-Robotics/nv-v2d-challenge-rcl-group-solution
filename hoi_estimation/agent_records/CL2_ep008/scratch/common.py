import numpy as np, json, trimesh
from scipy.spatial.transform import Rotation as Rot
INP='/work/inputs'
cam=json.load(open(f'{INP}/camera.json'))
K=np.array(cam['K']); Kp=np.array(cam['K_pipeline']); Twc=np.array(cam['T_world_cam'])
rec=np.load(f'{INP}/reconstruction.npz')
Tw=rec['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
def load_mesh():
    return trimesh.load(f'{INP}/object_mesh.glb',force='scene',process=False).to_geometry()
def rotdeg(Ra,Rb):
    R=np.swapaxes(Ra,-1,-2)@Rb
    c=(np.trace(R,axis1=-2,axis2=-1)-1)/2
    return np.degrees(np.arccos(np.clip(c,-1,1)))
