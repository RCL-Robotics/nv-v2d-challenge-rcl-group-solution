import sys, numpy as np, cv2, json, torch
sys.path.insert(0,'/work/outputs')
from render_utils import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
poses=np.load(sys.argv[1])[sys.argv[2]] if sys.argv[1]!='base' else np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
frames=[int(x) for x in sys.argv[3].split(',')]
out=sys.argv[4]
tiles=[]
for i in frames:
    im=cv2.imread('frames/%04d.jpg'%(i+1))
    T=np.linalg.inv(Twc)@poses[i]
    rgb,mask,_=R.render(T)
    rgb=(rgb[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mask=mask[0].cpu().numpy()
    ov=im.copy(); ov[mask]=(0.3*im[mask]+0.7*rgb[mask]).astype(np.uint8)
    cnt,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov,cnt,-1,(0,255,0),1)
    ys,xs=np.nonzero(mask) if mask.any() else (np.array([576]),np.array([768]))
    cx,cy=int(xs.mean()),int(ys.mean()); s=200
    x0,y0=max(0,cx-s),max(0,cy-s)
    crop=np.hstack([im[y0:y0+2*s,x0:x0+2*s],ov[y0:y0+2*s,x0:x0+2*s]])
    crop=cv2.resize(crop,(800,400)); cv2.putText(crop,str(i),(5,25),0,0.8,(0,255,255),2)
    tiles.append(crop)
while len(tiles)%2: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[j:j+2]) for j in range(0,len(tiles),2)]))
