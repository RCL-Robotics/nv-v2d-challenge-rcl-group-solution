import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from scipy.spatial.transform import Rotation as Rr
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
R=Renderer(v,f,uv,tex)
K=np.array([[800,0,128],[0,800,128],[0,0,1.]])
Ts=[]
# look along mesh axes: camera at distance 0.8
for rv in [[0,0,0],[0,90,0],[0,180,0],[0,270,0],[90,0,0],[-90,0,0]]:
    T=np.eye(4); T[:3,:3]=Rr.from_euler('xyz',rv,degrees=True).as_matrix(); T[2,3]=0.9
    Ts.append(T)
rgb,mask,_=R.render(np.stack(Ts),K,256,256)
ims=[(r.cpu().numpy()[...,::-1]*255).astype(np.uint8) for r in rgb]
cv2.imwrite('meshviews.jpg',np.hstack(ims))
# handle analysis: radial distance from z axis
r=np.linalg.norm(v[:,:2]-np.median(v[:,:2],0),axis=1)
print('median xy',np.median(v[:,:2],0))
far=v[r>np.percentile(r,97)]; print('far pts mean',far.mean(0), far.min(0), far.max(0))
for zq in np.linspace(v[:,2].min(),v[:,2].max(),11):
    s=v[np.abs(v[:,2]-zq)<0.005]
    if len(s): print('z %.3f'%zq, 'x[%.3f,%.3f] y[%.3f,%.3f]'%(s[:,0].min(),s[:,0].max(),s[:,1].min(),s[:,1].max()))
