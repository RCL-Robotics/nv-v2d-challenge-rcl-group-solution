import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rot
mesh=load_mesh('/work/inputs/object_mesh.glb')
K=np.array([[800,0,200],[0,800,250],[0,0,1.]])
R=Renderer(mesh,K,400,500,tex_max=4096)
tiles=[]
for yaw in [0,90,180,270]:
    # mesh z down in world -> camera y down: rotate mesh so z->cam y (down), then yaw about cam y
    R0=Rot.from_euler('x',-90,degrees=True).as_matrix()  # maps mesh z to -cam y? check
    Ry=Rot.from_euler('y',yaw,degrees=True).as_matrix()
    T=np.eye(4); T[:3,:3]=Ry@R0 if False else Ry@np.array([[1,0,0],[0,0,1],[0,-1,0]]).T
    T[:3,:3]=Ry@np.array([[1,0,0],[0,0,-1],[0,1,0]])  # mesh x->x, mesh z->y(down), mesh y->-z
    T[:3,3]=[0,0.05,2.2]
    Rx=Rot.from_euler('x',-20,degrees=True).as_matrix()
    T[:3,:3]=Rx@T[:3,:3]
    rgb,m,_,_=R.render(torch.tensor(T[None],dtype=torch.float32,device='cuda'))
    im=(rgb[0].cpu().numpy()*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.putText(im,f'yaw{yaw}',(5,20),0,0.6,(0,255,255),2)
    tiles.append(im)
cv2.imwrite('/work/scratch/exp/sides.jpg',np.hstack(tiles))
