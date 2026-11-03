import sys; sys.path.insert(0,'/work/scratch/s')
from rend import *
import cv2
from scipy.spatial.transform import Rotation as Rot
r=Renderer()
K=np.array([[800,0,320],[0,800,240],[0,0,1.]])
ims=[]
for ax,ang in [('y',0),('y',90),('y',180),('y',270),('x',90),('x',-90),('z',90),('z',-90)]:
    T=np.eye(4); T[:3,:3]=Rot.from_euler(ax,ang,degrees=True).as_matrix(); T[:3,3]=[-0.0,0,3.0]
    T[:3,3]-= T[:3,:3]@r.mesh.bounds.mean(0)
    m,d,c=r.render(T,K,480,640)
    im=(c[0].cpu().numpy()*255).astype(np.uint8)[...,::-1].copy()
    im[~m[0].cpu().numpy()]=80
    cv2.putText(im,f'{ax}{ang}',(10,30),0,1,(0,0,255),2); ims.append(im)
cv2.imwrite('/work/scratch/meshviews.jpg',np.vstack([np.hstack(ims[:4]),np.hstack(ims[4:])]))
