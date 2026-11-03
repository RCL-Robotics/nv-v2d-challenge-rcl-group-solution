import sys, numpy as np, json, cv2
sys.path.insert(0,'/work/outputs')
from render_utils import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
d=np.load(sys.argv[1]); i=int(sys.argv[2])
im=cv2.imread('frames/%04d.jpg'%(i+1)); tiles=[]
for T in [d['Tm'],d['Tsd'],d['Ts']]:
    rgb,mk,_=R.render(T); mk=mk[0].cpu().numpy(); rgb=(rgb[0].cpu().numpy()[...,::-1]*255).astype(np.uint8)
    ov=im.copy(); cnt,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,cnt,-1,(0,255,0),1)
    ys,xs=np.nonzero(mk); cx,cy=int(xs.mean()),int(ys.mean()); s=int(sys.argv[3])
    tiles.append(cv2.resize(ov[cy-s:cy+s,cx-s:cx+s],(512,512),interpolation=cv2.INTER_CUBIC))
    tiles.append(cv2.resize(rgb[cy-s:cy+s,cx-s:cx+s],(512,512)))
cv2.imwrite(sys.argv[4],np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
