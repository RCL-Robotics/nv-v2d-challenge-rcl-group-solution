import numpy as np, json, h5py, trimesh
INP='/work/inputs'
def load():
    r=dict(np.load(INP+'/reconstruction.npz'))
    cam=json.load(open(INP+'/camera.json'))
    return r,cam
def mesh():
    return trimesh.load(INP+'/object_mesh.glb',force='scene',process=False).to_geometry()
def rotang(R1,R2):
    c=np.trace(np.swapaxes(R1,-1,-2)@R2,axis1=-2,axis2=-1)
    return np.degrees(np.arccos(np.clip((c-1)/2,-1,1)))
