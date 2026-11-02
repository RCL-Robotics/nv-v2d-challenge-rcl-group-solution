import sys, numpy as np, cv2, json, h5py
sys.path.insert(0,'/work/outputs')
from render_utils import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
src=sys.argv[1]; key=sys.argv[2]
poses=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'] if src=='base' else np.load(src)[key]
f=h5py.File('/work/inputs/masks.h5')
frames=[int(x) for x in sys.argv[3].split(',')]
tiles=[]
for i in frames:
    im=cv2.imread('frames/%04d.jpg'%(i+1))
    _,mask,_=R.render(np.linalg.inv(Twc)@poses[i],need_rgb=False); mask=mask[0].cpu().numpy()
    mo=f['object'][i]; mp=f['person'][i]
    ov=im.copy()
    ov[mo]=(0.5*ov[mo]+[0,0,120]).astype(np.uint8)
    cnt,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov,cnt,-1,(0,255,0),1)
    cnt,_=cv2.findContours(mp.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov,cnt,-1,(255,0,0),1)
    ys,xs=np.nonzero(mo|mask); cx,cy=int(xs.mean()),int(ys.mean()); s=int(sys.argv[5]) if len(sys.argv)>5 else 160
    x0,y0=max(0,cx-s),max(0,cy-s)
    crop=cv2.resize(ov[y0:y0+2*s,x0:x0+2*s],(480,480),interpolation=cv2.INTER_NEAREST); cv2.putText(crop,str(i),(5,25),0,0.8,(0,255,255),2)
    tiles.append(crop)
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[4],np.vstack([np.hstack(tiles[j:j+3]) for j in range(0,len(tiles),3)]))
