import numpy as np, json, trimesh, h5py
INP='/work/inputs'
def load():
    r=np.load(f'{INP}/reconstruction.npz')
    cam=json.load(open(f'{INP}/camera.json'))
    pl=np.array(json.load(open(f'{INP}/ground_plane.json'))['plane'])
    m=trimesh.load(f'{INP}/object_mesh.glb',force='scene',process=False).to_geometry()
    return r,cam,pl,m
def rotang(R1,R2):
    R=R1.swapaxes(-1,-2)@R2
    return np.degrees(np.arccos(np.clip((np.trace(R,axis1=-2,axis2=-1)-1)/2,-1,1)))
