import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch')
from hoi_render import *
from hoi_fit import *
import cv2, json, h5py
# args: poses.npy (F,4,4 world) out.jpg frames
Tw=np.load(sys.argv[1]); out=sys.argv[2]; frames=[int(x) for x in sys.argv[3].split(',')]
Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,1152,1536)
f=h5py.File('/work/inputs/masks.h5','r')
tiles=[]
for i in frames:
    im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')
    ov=im.copy()
    for T,col in [(Tb[i],(0,0,255)),(Tw[i],(0,255,0))]:
        _,mk,_=R.render(np.linalg.inv(Twc)@T,K); mk=mk[0].cpu().numpy().astype(np.uint8)
        cnt,_=cv2.findContours(mk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,cnt,-1,col,1)
    o=f['object'][i]; cnt,_=cv2.findContours(o.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,cnt,-1,(255,255,0),1)
    rgb,mk,_=R.render(np.linalg.inv(Twc)@Tw[i],K); rgb=(rgb[0].cpu().numpy()*255).astype(np.uint8)[...,::-1]
    x0,y0,w,h=CROP
    t=np.hstack([ov[y0:y0+h,x0:x0+w],rgb[y0:y0+h,x0:x0+w]]); cv2.putText(t,str(i),(5,25),0,0.8,(0,0,255),2); tiles.append(t)
ncol=int(sys.argv[4]) if len(sys.argv)>4 else 2
while len(tiles)%ncol: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[i:i+ncol]) for i in range(0,len(tiles),ncol)]))
