import sys; sys.path.insert(0,'/work/outputs')
from common import *
from scipy.spatial.transform import Rotation as R
m=load_mesh('/work/inputs/object_mesh.glb')
K=np.array([[2400,0,300],[0,2400,400],[0,0,1.]])
r=Renderer(m,K,H=800,W=600)
ims=[]
for yaw in range(0,360,45):
    Rb=np.array([[1,0,0],[0,0,-1],[0,1,0.]])  # mesh z -> cam -y (up), mesh y -> cam z
    Ry=R.from_euler('y',yaw,degrees=True).as_matrix()
    T=np.eye(4); T[:3,:3]=Ry@Rb; T[:3,3]=[0,0,2.2]
    mk,rgb,d,_=r.render(T[None])
    im=(rgb[0].cpu().numpy()*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.putText(im,f'yaw{yaw}',(10,30),0,1,(0,0,255),2); ims.append(im)
cv2.imwrite('views2.jpg', np.vstack([np.hstack(ims[:4]),np.hstack(ims[4:])]))
