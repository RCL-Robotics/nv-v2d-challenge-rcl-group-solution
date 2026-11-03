import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rr
R=MeshRenderer('/work/inputs/object_mesh.glb')
K=np.array([[800,0,200],[0,800,200],[0,0,1.]])
outs=[]
for name,rv in [('+y view',[90,0,0]),('-y view',[-90,0,0]),('+x',[0,-90,0]),('-x',[0,90,0]),('+z',[0,180,0]),('-z',[0,0,0])]:
    T=np.eye(4); T[:3,:3]=Rr.from_euler('xyz',rv,degrees=True).as_matrix(); T[2,3]=0.8
    rgb,m,_=R.render(T[None],K,(0,0,400,400),(400,400))
    im=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8).copy()
    cv2.putText(im,name,(5,25),0,0.8,(0,0,255),2); outs.append(im)
cv2.imwrite('views.jpg',np.hstack(outs))
