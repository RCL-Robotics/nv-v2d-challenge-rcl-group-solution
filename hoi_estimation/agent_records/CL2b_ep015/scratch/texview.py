import sys; sys.path.insert(0,'/work/outputs')
from common import *
m=load_mesh('/work/inputs/object_mesh.glb')
tex=np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
cv2.imwrite('tex.jpg', cv2.resize(tex[:,:,::-1],(1000,1000)))
K=np.array([[1500,0,400],[0,1500,400],[0,0,1.]])
r=Renderer(m,K,H=800,W=800)
from scipy.spatial.transform import Rotation as R
ims=[]
for yaw in [0,90,180,270]:
    # camera looking at object from horizontal; mesh z up -> cam -y
    Rb=R.from_euler('x',-90,degrees=True).as_matrix()  # mesh z -> cam -y
    Ry=R.from_euler('y',yaw,degrees=True).as_matrix()
    T=np.eye(4); T[:3,:3]=Ry@Rb; T[:3,3]=[0,0,2.0]
    mk,rgb,d,_=r.render(T[None])
    im=(rgb[0].cpu().numpy()*255).astype(np.uint8)[:,:,::-1].copy().copy()
    cv2.putText(im,f'yaw{yaw}',(10,30),0,1,(0,0,255),2); ims.append(im)
cv2.imwrite('views.jpg', np.hstack(ims))
