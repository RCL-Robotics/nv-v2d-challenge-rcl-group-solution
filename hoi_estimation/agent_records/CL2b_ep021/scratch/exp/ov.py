import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s
R=Renderer(mesh,Ks,int(W*s),int(H*s))
Tc=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
# check projection
v=np.asarray(mesh.vertices); i=0
vc=(Tc[i][:3,:3]@v.T).T+Tc[i][:3,3]; p=(Ks@vc.T).T; p=p[:,:2]/p[:,2:]
o=R.render(Tc[i:i+1]); m=o['mask'][0].cpu().numpy(); ys,xs=np.nonzero(m)
print('proj centroid',p.mean(0),'render bbox',xs.min(),xs.max(),ys.min(),ys.max(),'proj bbox',p.min(0),p.max(0))
import h5py; fm=h5py.File('/work/inputs/masks.h5','r')
frames=[int(a) for a in sys.argv[1].split(',')]
tiles=[]
for i in frames:
    img=cv2.resize(cv2.imread(f'/work/scratch/frames/{i:04d}.jpg'),(int(W*s),int(H*s)))
    o=R.render(Tc[i:i+1]); m=o['mask'][0].cpu().numpy(); c=(o['color'][0].cpu().numpy()[...,::-1]*255)
    om=cv2.resize(fm['object'][i].astype(np.uint8),(int(W*s),int(H*s)))>0
    ov=img.copy(); ov[m]=(0.5*ov[m]+0.5*c[m]).astype(np.uint8)
    cnt,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,cnt,-1,(0,255,0),1)
    ys,xs=np.nonzero(m|om); 
    if len(xs)==0: x0,y0=0,0
    else: x0=max(0,int(xs.mean())-130); y0=max(0,int(ys.mean())-170)
    t=ov[y0:y0+340,x0:x0+260]; t=cv2.copyMakeBorder(t,0,340-t.shape[0],0,260-t.shape[1],0)
    cv2.putText(t,str(i),(5,20),0,0.6,(0,0,255),2); tiles.append(t)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
g=np.vstack([np.hstack(tiles[k:k+6]) for k in range(0,len(tiles),6)])
cv2.imwrite(sys.argv[2],g)
