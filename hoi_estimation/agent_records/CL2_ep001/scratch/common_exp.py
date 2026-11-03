import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from fit import *
cam,K,Twc,rec=load_inputs('/work/inputs')
FULL=np.load('/work/scratch/cache/full.npy',mmap_mode='r')
OBJ=np.load('/work/scratch/cache/obj.npy',mmap_mode='r')
PER=np.load('/work/scratch/cache/person.npy',mmap_mode='r')
CHAIR=np.load('/work/scratch/cache/chair.npy')
Tb=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(np.float64)
Tfp=rec['T_cam_mesh_foundationpose'].astype(np.float64); Tcn=rec['T_cam_mesh_coconet'].astype(np.float64)
kern=np.ones((7,7),np.uint8)
def ignore_mask(i):
    return cv2.dilate((PER[i]|CHAIR).astype(np.uint8),kern)>0
def make_fd(fit,ids,Tinit,R=192):
    boxes=[]
    for i,T in zip(ids,Tinit):
        corners=trimesh.bounds.corners(fit.mesh.bounds)
        uv=project(K,T,corners)
        boxes.append(box_from(OBJ[i],uv,pad=0.15))
    return FrameData([FULL[i] for i in ids],[OBJ[i] for i in ids],[ignore_mask(i) for i in ids],boxes,K,R)
