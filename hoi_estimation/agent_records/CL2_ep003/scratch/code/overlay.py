import sys; sys.path.insert(0,'/work/outputs')
from lib import *
import numpy as np, cv2, h5py
key=sys.argv[1]; out=sys.argv[2]; idx=[int(x) for x in sys.argv[3].split(',')]
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
cam,K,Twc,rec=load_inputs('/work/inputs')
if key=='base': Tc=np.linalg.inv(Twc)[None]@rec['T_world_mesh']; KK=K
elif key.endswith('.npz'): Tc=np.linalg.inv(Twc)[None]@np.load(key)['T_world_mesh']; KK=K
else: Tc=rec[key]; KK=np.array(cam['K_pipeline'])
R=Renderer(v,f,uv,tex)
cap=cv2.VideoCapture('/work/inputs/video.mp4'); i=0; tiles=[]
while i<=max(idx):
    ok,im=cap.read()
    if i in idx:
        rgb,mask,_=R.render(Tc[i:i+1],KK,W,H)
        rgb=(rgb[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mask[0].cpu().numpy()
        ys,xs=np.nonzero(mk); cy,cx=(int(ys.mean()),int(xs.mean())) if len(ys) else (500,800)
        h=80; sl=(slice(max(cy-h,0),cy+h),slice(max(cx-h,0),cx+h))
        a=im[sl].copy(); b=im[sl].copy(); b[mk[sl]]=(0.3*b[mk[sl]]+0.7*rgb[sl][mk[sl]]).astype(np.uint8)
        c=np.hstack([a,b]); c=cv2.resize(c,(640,320)); cv2.putText(c,str(i),(5,25),0,0.8,(0,0,255),2); tiles.append(c)
    i+=1
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
