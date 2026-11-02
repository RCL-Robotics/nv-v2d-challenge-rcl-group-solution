import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from scipy.spatial.transform import Rotation as R
K=np.array([[800,0,200],[0,800,300],[0,0,1.]])
rd=Renderer('/work/inputs/object_mesh.glb',K,600,400)
c=rd.mesh.bounds.mean(0)
ims=[]
for ax,ang in [('y',0),('y',90),('y',180),('y',270),('x',90),('x',-90)]:
    Rm=R.from_euler(ax,ang,degrees=True).as_matrix()
    T=np.eye(4); T[:3,:3]=Rm; T[:3,3]=-Rm@c+np.array([0,0,3.0])
    o=rd.render(T[None]); im=(o['color'][0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8)
    im[~o['mask'][0].cpu().numpy()]=255
    cv2.putText(im,f'{ax}{ang}',(5,20),0,0.6,(0,0,0),2); ims.append(im)
cv2.imwrite('/work/scratch/viz/views.png',np.concatenate(ims,1))
